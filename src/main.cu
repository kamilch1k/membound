// membound — hand-written FP16 GEMV kernels pushed toward the memory-bandwidth limit.
//
// At batch size 1, LLM decode is GEMV: every output token streams each weight
// matrix from DRAM once and does ~1 FMA per weight. The kernel is memory-bound,
// so the metric that counts is MBU (memory-bandwidth utilization) = achieved
// GB/s / the card's *measured* achievable GB/s — not the datasheet number.
//
// Stages (each is a kernel below, benchmarked against the same measured ceiling):
//   0. probes  — copy + pure-read streaming kernels establish the real ceiling
//   1. naive   — one thread per row (uncoalesced row walk; the "before")
//   2. warp    — one warp per row: coalesced loads + shared-memory x tile + shuffle reduce
//   3. vec     — 128-bit (float4 = 8 half) loads, FP32 accumulation
//
// Correctness: every kernel is checked against a double-precision CPU reference
// on every run before it is timed. No benchmark without a passing check.

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

// ---------------------------------------------------------------- probes ----

__global__ void copy_kernel(const float4* __restrict__ src, float4* __restrict__ dst, size_t n4) {
    size_t i      = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (; i < n4; i += stride) dst[i] = src[i];
}

// Pure-read probe: GEMV traffic is ~100% reads (M*N*2 bytes read, 4*M written),
// so the read ceiling is the honest MBU denominator. The impossible-value guard
// keeps the compiler from eliding the loads.
__global__ void read_kernel(const float4* __restrict__ src, float* __restrict__ sink, size_t n4) {
    size_t i      = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    float acc = 0.f;
    for (; i < n4; i += stride) {
        float4 v = src[i];
        acc += v.x + v.y + v.z + v.w;
    }
    if (acc == -1.2345678f) sink[0] = acc;
}

// -------------------------------------------------------- stage 1: naive ----

// One thread per output row. Adjacent threads read rows N elements apart, so
// every warp-level load touches 32 different cache lines: maximally uncoalesced.
__global__ void gemv_naive(const __half* __restrict__ A, const __half* __restrict__ x,
                           float* __restrict__ y, int M, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) return;
    const __half* a = A + (size_t)row * N;
    float acc = 0.f;
    for (int j = 0; j < N; ++j)
        acc = fmaf(__half2float(a[j]), __half2float(x[j]), acc);
    y[row] = acc;
}

// --------------------------------------------------------- stage 2: warp ----

// One warp per row: the 32 lanes read consecutive elements -> coalesced.
// x is staged through shared memory in tiles and reused by every warp in the
// block; partial sums are combined with a warp shuffle reduction.
constexpr int X_TILE = 2048; // halves; 4 KB of shared memory per block

__global__ void gemv_warp(const __half* __restrict__ A, const __half* __restrict__ x,
                          float* __restrict__ y, int M, int N) {
    __shared__ __half xs[X_TILE];
    const int lane = threadIdx.x;                       // 0..31
    const int warp = threadIdx.y;
    const int row  = blockIdx.x * blockDim.y + warp;

    float acc = 0.f;
    for (int t = 0; t < N; t += X_TILE) {
        const int len = min(X_TILE, N - t);
        for (int i = warp * 32 + lane; i < len; i += blockDim.y * 32)
            xs[i] = x[t + i];
        __syncthreads();
        if (row < M) {
            const __half* a = A + (size_t)row * N + t;
            for (int j = lane; j < len; j += 32)
                acc = fmaf(__half2float(a[j]), __half2float(xs[j]), acc);
        }
        __syncthreads();
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (row < M && lane == 0) y[row] = acc;
}

// ---------------------------------------------------------- stage 3: vec ----

// Same warp-per-row shape, but 128-bit loads: each lane reads a float4
// (8 halves) of A and x per step. Fewer, full-width memory transactions.
// Requires N % 8 == 0 (checked on the host).
__global__ void gemv_vec(const __half* __restrict__ A, const __half* __restrict__ x,
                         float* __restrict__ y, int M, int N) {
    const int lane = threadIdx.x;
    const int row  = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= M) return;

    const float4* a4 = reinterpret_cast<const float4*>(A + (size_t)row * N);
    const float4* x4 = reinterpret_cast<const float4*>(x);
    const int n8 = N / 8;

    float acc = 0.f;
    for (int i = lane; i < n8; i += 32) {
        const float4 av = __ldg(&a4[i]);
        const float4 xv = __ldg(&x4[i]);
        const __half2* ah = reinterpret_cast<const __half2*>(&av);
        const __half2* xh = reinterpret_cast<const __half2*>(&xv);
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            const float2 af = __half22float2(ah[k]);
            const float2 xf = __half22float2(xh[k]);
            acc = fmaf(af.x, xf.x, acc);
            acc = fmaf(af.y, xf.y, acc);
        }
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) y[row] = acc;
}

// -------------------------------------------------------------- harness ----

struct Timing { float ms; };

template <typename Launch>
Timing time_kernel(Launch&& launch, int warmup = 20, int iters = 100) {
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

// max relative error vs the double-precision reference
static double check(const std::vector<float>& got, const std::vector<double>& ref) {
    double worst = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const double rel = std::abs(got[i] - ref[i]) / (std::abs(ref[i]) + 1.0);
        worst = std::max(worst, rel);
    }
    return worst;
}

int main() {
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("device: %s  (sm_%d%d, %d SMs, %.1f GB)\n",
                prop.name, prop.major, prop.minor, prop.multiProcessorCount,
                prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));

    // ---- ceiling probes (256 MiB working set) ----
    const size_t probe_bytes = 256ull << 20;
    const size_t n4 = probe_bytes / sizeof(float4);
    float4 *psrc, *pdst;
    float* psink;
    CUDA_CHECK(cudaMalloc(&psrc, probe_bytes));
    CUDA_CHECK(cudaMalloc(&pdst, probe_bytes));
    CUDA_CHECK(cudaMalloc(&psink, sizeof(float)));
    CUDA_CHECK(cudaMemset(psrc, 1, probe_bytes));

    const int pblocks = prop.multiProcessorCount * 8;
    Timing tc = time_kernel([&] { copy_kernel<<<pblocks, 256>>>(psrc, pdst, n4); });
    Timing tr = time_kernel([&] { read_kernel<<<pblocks, 256>>>(psrc, psink, n4); });
    CUDA_CHECK(cudaGetLastError());
    const double copy_gbs = 2.0 * probe_bytes / (tc.ms * 1e-3) / 1e9;
    const double read_gbs = 1.0 * probe_bytes / (tr.ms * 1e-3) / 1e9;
    std::printf("measured ceiling:  copy %.1f GB/s   pure-read %.1f GB/s   (MBU denominator = pure-read)\n\n",
                copy_gbs, read_gbs);
    CUDA_CHECK(cudaFree(psrc));
    CUDA_CHECK(cudaFree(pdst));
    CUDA_CHECK(cudaFree(psink));

    // ---- GEMV shapes: typical decode projections (Llama-ish) ----
    struct Shape { int M, N; };
    const Shape shapes[] = {{4096, 4096}, {8192, 8192}, {14336, 4096}, {4096, 14336}};

    std::printf("%-12s %-7s %10s %10s %8s %10s   %s\n",
                "shape", "kernel", "ms", "GB/s", "MBU%", "GFLOP/s", "max-rel-err");

    for (const Shape s : shapes) {
        const int M = s.M, N = s.N;
        if (N % 8 != 0) { std::fprintf(stderr, "N must be a multiple of 8\n"); return 1; }

        // deterministic input (same seed every run — reproducible numbers)
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(-1.f, 1.f);
        std::vector<__half> hA((size_t)M * N), hx(N);
        for (auto& v : hA) v = __float2half(dist(rng));
        for (auto& v : hx) v = __float2half(dist(rng));

        // double-precision reference on the *rounded* half values
        std::vector<double> ref(M, 0.0);
        for (int i = 0; i < M; ++i) {
            double acc = 0.0;
            const __half* a = hA.data() + (size_t)i * N;
            for (int j = 0; j < N; ++j)
                acc += (double)__half2float(a[j]) * (double)__half2float(hx[j]);
            ref[i] = acc;
        }

        __half *dA, *dx;
        float* dy;
        CUDA_CHECK(cudaMalloc(&dA, (size_t)M * N * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&dx, (size_t)N * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&dy, (size_t)M * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(dA, hA.data(), (size_t)M * N * sizeof(__half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dx, hx.data(), (size_t)N * sizeof(__half), cudaMemcpyHostToDevice));

        const double bytes  = (double)M * N * 2 + N * 2 + M * 4;
        const double flops  = 2.0 * M * N;
        std::vector<float> out(M);

        struct K { const char* name; void (*run)(const __half*, const __half*, float*, int, int, cudaStream_t); };
        auto bench = [&](const char* name, auto&& launch) {
            launch(); // once for correctness
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(out.data(), dy, M * sizeof(float), cudaMemcpyDeviceToHost));
            const double err = check(out, ref);
            if (err > 1e-2) {
                std::fprintf(stderr, "%s FAILED correctness: max rel err %.3g\n", name, err);
                std::exit(1);
            }
            const Timing t = time_kernel(launch);
            const double gbs = bytes / (t.ms * 1e-3) / 1e9;
            std::printf("%-12s %-7s %10.4f %10.1f %8.1f %10.1f   %.2e\n",
                        (std::to_string(M) + "x" + std::to_string(N)).c_str(), name,
                        t.ms, gbs, 100.0 * gbs / read_gbs, flops / (t.ms * 1e-3) / 1e9, err);
        };

        bench("naive", [&] { gemv_naive<<<(M + 255) / 256, 256>>>(dA, dx, dy, M, N); });
        bench("warp",  [&] { gemv_warp<<<(M + 7) / 8, dim3(32, 8)>>>(dA, dx, dy, M, N); });
        bench("vec",   [&] { gemv_vec<<<(M + 7) / 8, dim3(32, 8)>>>(dA, dx, dy, M, N); });

        CUDA_CHECK(cudaFree(dA));
        CUDA_CHECK(cudaFree(dx));
        CUDA_CHECK(cudaFree(dy));
        std::printf("\n");
    }

    std::printf("done. all kernels passed the double-precision correctness check.\n");
    return 0;
}
