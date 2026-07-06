// membound softmax — row-wise FP16 softmax, naive 3-pass baseline vs a fused
// online kernel (Milakov & Gimelshein running max+sum), benchmarked against the
// measured *copy* ceiling: softmax's floor traffic is one read + one write per
// element, a balanced read/write stream — so pure-read would be the wrong
// denominator here (GEMV's traffic is ~100% reads; softmax's is 50/50).
//
// Correctness: every kernel is checked against a double-precision CPU
// reference on every run before it is timed.

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <vector>
#include <random>
#include <string>
#include <algorithm>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err__ = (call);                                             \
        if (err__ != cudaSuccess) {                                             \
            std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n",                \
                         cudaGetErrorName(err__), __FILE__, __LINE__,           \
                         cudaGetErrorString(err__));                            \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

__global__ void copy_kernel(const float4* __restrict__ src, float4* __restrict__ dst, size_t n4) {
    size_t i      = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (; i < n4; i += stride) dst[i] = src[i];
}

// ------------------------------------------------------------- baseline ----

// One thread per row, three scalar passes (max, sum, normalize). Uncoalesced
// AND reads the row from DRAM three times: the "before".
__global__ void softmax_naive(const __half* __restrict__ in, __half* __restrict__ out,
                              int M, int N) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) return;
    const __half* x = in + (size_t)row * N;
    __half*       y = out + (size_t)row * N;
    float m = -1e30f;
    for (int j = 0; j < N; ++j) m = fmaxf(m, __half2float(x[j]));
    float d = 0.f;
    for (int j = 0; j < N; ++j) d += __expf(__half2float(x[j]) - m);
    const float inv = 1.f / d;
    for (int j = 0; j < N; ++j)
        y[j] = __float2half(__expf(__half2float(x[j]) - m) * inv);
}

// ---------------------------------------------------------------- fused ----

// One block per row, 128-bit loads. Pass 1 computes max AND sum in a single
// sweep with the online update (new x: m' = max(m,x); d = d*exp(m-m') +
// exp(x-m')), so the max pass costs no extra read. Pass 2 normalizes; its
// re-read of the row usually hits L2 (the row was just streamed through it),
// so DRAM traffic approaches the 1-read + 1-write floor.
constexpr int SM_BLOCK = 256;

__global__ void softmax_online(const __half* __restrict__ in, __half* __restrict__ out,
                               int M, int N) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const float4* x4 = reinterpret_cast<const float4*>(in + (size_t)row * N);
    const int n8 = N / 8;

    float m = -1e30f, d = 0.f;
    for (int i = tid; i < n8; i += SM_BLOCK) {
        const float4 v = x4[i];
        const __half2* h = reinterpret_cast<const __half2*>(&v);
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            const float2 f = __half22float2(h[k]);
            float nm = fmaxf(m, f.x);
            d = d * __expf(m - nm) + __expf(f.x - nm);
            m = nm;
            nm = fmaxf(m, f.y);
            d = d * __expf(m - nm) + __expf(f.y - nm);
            m = nm;
        }
    }

    // block-level reduction of the (max, sum) pairs
    __shared__ float sm[SM_BLOCK], sd[SM_BLOCK];
    sm[tid] = m; sd[tid] = d;
    __syncthreads();
    for (int s = SM_BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) {
            const float m2 = sm[tid + s], d2 = sd[tid + s];
            const float nm = fmaxf(sm[tid], m2);
            sd[tid] = sd[tid] * __expf(sm[tid] - nm) + d2 * __expf(m2 - nm);
            sm[tid] = nm;
        }
        __syncthreads();
    }
    const float gm  = sm[0];
    const float inv = 1.f / sd[0];

    float4* y4 = reinterpret_cast<float4*>(out + (size_t)row * N);
    for (int i = tid; i < n8; i += SM_BLOCK) {
        const float4 v = x4[i];
        const __half2* h = reinterpret_cast<const __half2*>(&v);
        float4 w;
        __half2* g = reinterpret_cast<__half2*>(&w);
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            const float2 f = __half22float2(h[k]);
            g[k] = __floats2half2_rn(__expf(f.x - gm) * inv, __expf(f.y - gm) * inv);
        }
        y4[i] = w;
    }
}

// -------------------------------------------------------------- harness ----

struct Timing { float ms; };

template <typename Launch>
Timing time_kernel(Launch&& launch, int warmup = 10, int iters = 50) {
    for (int i = 0; i < warmup; ++i) launch();
    cudaEvent_t beg, end;
    CUDA_CHECK(cudaEventCreate(&beg));
    CUDA_CHECK(cudaEventCreate(&end));
    CUDA_CHECK(cudaEventRecord(beg));
    for (int i = 0; i < iters; ++i) launch();
    CUDA_CHECK(cudaEventRecord(end));
    CUDA_CHECK(cudaEventSynchronize(end));
    float total = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&total, beg, end));
    CUDA_CHECK(cudaEventDestroy(beg));
    CUDA_CHECK(cudaEventDestroy(end));
    return {total / iters};
}

int main(int argc, char** argv) {
    const bool profile_mode = argc > 1 && std::string(argv[1]) == "--profile";
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("device: %s  (sm_%d%d, %d SMs, %.1f GB)\n",
                prop.name, prop.major, prop.minor, prop.multiProcessorCount,
                prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));

    // copy ceiling = the denominator (balanced read+write, like softmax itself);
    // measured before and after the suite, max wins (laptop clock drift)
    auto copy_ceiling = [&]() -> double {
        const size_t probe_bytes = 256ull << 20;
        const size_t n4 = probe_bytes / sizeof(float4);
        float4 *psrc, *pdst;
        CUDA_CHECK(cudaMalloc(&psrc, probe_bytes));
        CUDA_CHECK(cudaMalloc(&pdst, probe_bytes));
        CUDA_CHECK(cudaMemset(psrc, 1, probe_bytes));
        const int pblocks = prop.multiProcessorCount * 8;
        Timing tc = time_kernel([&] { copy_kernel<<<pblocks, 256>>>(psrc, pdst, n4); });
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaFree(psrc));
        CUDA_CHECK(cudaFree(pdst));
        return 2.0 * probe_bytes / (tc.ms * 1e-3) / 1e9;
    };
    double copy_gbs = 1.0;
    if (!profile_mode) copy_gbs = copy_ceiling();

    struct Shape { int M, N; };
    const Shape shapes[] = {{8192, 8192}, {4096, 32768}, {1024, 131072}};

    struct Row { int M, N; const char* name; float ms; double gbs, err; };
    std::vector<Row> rows;

    for (const Shape s : shapes) {
        const int M = s.M, N = s.N;

        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(-4.f, 4.f); // logit-ish range
        std::vector<__half> hin((size_t)M * N);
        for (auto& v : hin) v = __float2half(dist(rng));

        // double-precision reference on the rounded half values
        std::vector<double> ref((size_t)M * N);
        for (int i = 0; i < M; ++i) {
            const __half* x = hin.data() + (size_t)i * N;
            double m = -1e300;
            for (int j = 0; j < N; ++j) m = std::max(m, (double)__half2float(x[j]));
            double d = 0.0;
            for (int j = 0; j < N; ++j) d += std::exp((double)__half2float(x[j]) - m);
            for (int j = 0; j < N; ++j)
                ref[(size_t)i * N + j] = std::exp((double)__half2float(x[j]) - m) / d;
        }

        // rotate >=256MB of input copies so iterations stream cold from DRAM
        const size_t bytesIn = (size_t)M * N * sizeof(__half);
        const int    ncopies = (int)std::max<size_t>(1, ((256ull << 20) + bytesIn - 1) / bytesIn);
        __half *din, *dout;
        CUDA_CHECK(cudaMalloc(&din, bytesIn * ncopies));
        CUDA_CHECK(cudaMalloc(&dout, bytesIn));
        for (int c = 0; c < ncopies; ++c)
            CUDA_CHECK(cudaMemcpy(din + c * (size_t)M * N, hin.data(), bytesIn, cudaMemcpyHostToDevice));
        size_t iter = 0;
        auto nextIn = [&]() -> const __half* { return din + (iter++ % ncopies) * (size_t)M * N; };

        // floor traffic: read every element once + write every element once
        const double bytes = 2.0 * (double)M * N * sizeof(__half);
        std::vector<float> out((size_t)M * N);

        auto bench = [&](const char* name, auto&& launch) {
            launch();
            CUDA_CHECK(cudaGetLastError());
            std::vector<__half> hout((size_t)M * N);
            CUDA_CHECK(cudaMemcpy(hout.data(), dout, bytesIn, cudaMemcpyDeviceToHost));
            // FP16 outputs below ~6e-5 are subnormal — their *relative* precision
            // collapses by construction (a 4e-8 probability rounds to the nearest
            // 6e-8 step). Score error relative to max(ref, 1e-4): full relative
            // strictness for every probability that matters, absolute-error
            // strictness below half's precision floor.
            double worst = 0.0;
            for (size_t i = 0; i < ref.size(); ++i) {
                const double rel = std::abs((double)__half2float(hout[i]) - ref[i]) / std::max(ref[i], 1e-4);
                worst = std::max(worst, rel);
            }
            if (worst > 5e-2) {
                std::fprintf(stderr, "%s FAILED correctness: max rel err %.3g\n", name, worst);
                std::exit(1);
            }
            if (profile_mode) {
                std::printf("%dx%d %s: correct (max rel err %.2e), launched once for ncu\n",
                            M, N, name, worst);
                return;
            }
            const Timing t = time_kernel(launch);
            rows.push_back({M, N, name, t.ms, bytes / (t.ms * 1e-3) / 1e9, worst});
        };

        bench("naive",  [&] { softmax_naive<<<(M + 255) / 256, 256>>>(nextIn(), dout, M, N); });
        bench("online", [&] { softmax_online<<<M, SM_BLOCK>>>(nextIn(), dout, M, N); });

        CUDA_CHECK(cudaFree(din));
        CUDA_CHECK(cudaFree(dout));
        if (profile_mode) std::printf("\n");
    }

    if (!profile_mode) {
        copy_gbs = std::max(copy_gbs, copy_ceiling());
        std::printf("copy ceiling (max of before/after): %.1f GB/s  = MBU denominator (read+write stream)\n\n", copy_gbs);
        std::printf("%-12s %-7s %10s %10s %8s   %s\n",
                    "shape", "kernel", "ms", "GB/s", "MBU%", "max-rel-err");
        long long prevShape = 0;
        for (const Row& r : rows) {
            const long long shape = (long long)r.M << 20 | r.N;
            if (prevShape != 0 && shape != prevShape) std::printf("\n");
            prevShape = shape;
            std::printf("%-12s %-7s %10.4f %10.1f %8.1f   %.2e\n",
                        (std::to_string(r.M) + "x" + std::to_string(r.N)).c_str(), r.name,
                        r.ms, r.gbs, 100.0 * r.gbs / copy_gbs, r.err);
        }
    }

    std::printf("\ndone. all kernels passed the double-precision correctness check.\n");
    return 0;
}
