// membound dequant — quantized-weight GEMV: INT8 and INT4 weights streamed
// from DRAM, dequantized in-register, FP32 accumulate against an FP16
// activation vector.
//
// This is the kernel batch-1 decode actually runs in production: once GEMV is
// at the bandwidth limit (see main.cu), the only way to go faster is to move
// fewer bytes — quantization converts bandwidth headroom directly into
// tokens/s. INT8 halves the weight traffic of FP16; INT4 quarters it. If the
// dequant kernels hold the same MBU as the FP16 kernel, tokens/s scales with
// the compression ratio. That's the claim this benchmark tests: the fp16 `vec`
// kernel runs in the same binary as the baseline, and each quantized kernel
// reports its measured speedup over it.
//
// Formats (deliberately standard-shaped):
//   q8: symmetric per-row scale.  a ≈ q * s_row,   q ∈ [-127,127],  1 B/elem
//   q4: symmetric per-group-of-128 scales (AWQ/GGUF-style granularity).
//       a ≈ q * s_group, q ∈ [-7,7] stored as (q+8) in a nibble, 0.53 B/elem
//
// Correctness: the double-precision reference is computed on the *dequantized*
// weights (q·s exactly), so the check verifies the kernel's arithmetic and
// indexing — quantization error itself is a modeling choice, not a bug, and
// is reported separately as quant-rmse.

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

// ------------------------------------------------- fp16 baseline (vec) ----

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

// ------------------------------------------------------------------ q8 ----

// Warp per row; each lane streams 16 int8 weights per 128-bit load (a full
// row of FP16 x costs 2x the weight bytes here — x is hot in L2, weights are
// the cold stream). Dequant is one int->float convert folded into the FMA.
__global__ void gemv_q8(const int8_t* __restrict__ A, const float* __restrict__ rowScale,
                        const __half* __restrict__ x, float* __restrict__ y, int M, int N) {
    const int lane = threadIdx.x;
    const int row  = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= M) return;
    const int4*   a16 = reinterpret_cast<const int4*>(A + (size_t)row * N); // 16 int8
    const float4* x4  = reinterpret_cast<const float4*>(x);                 // 8 halves
    const int n16 = N / 16;
    float acc = 0.f;
    for (int i = lane; i < n16; i += 32) {
        const int4 av = __ldg(&a16[i]);
        const float4 xv0 = __ldg(&x4[2 * i]);
        const float4 xv1 = __ldg(&x4[2 * i + 1]);
        const int8_t*  aq  = reinterpret_cast<const int8_t*>(&av);
        const __half2* xh0 = reinterpret_cast<const __half2*>(&xv0);
        const __half2* xh1 = reinterpret_cast<const __half2*>(&xv1);
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            const float2 f0 = __half22float2(xh0[k]);
            acc = fmaf((float)aq[2 * k],     f0.x, acc);
            acc = fmaf((float)aq[2 * k + 1], f0.y, acc);
            const float2 f1 = __half22float2(xh1[k]);
            acc = fmaf((float)aq[8 + 2 * k],     f1.x, acc);
            acc = fmaf((float)aq[8 + 2 * k + 1], f1.y, acc);
        }
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) y[row] = acc * rowScale[row];
}

// ------------------------------------------------------------------ q4 ----

// Two weights per byte; each lane's 128-bit load covers 32 elements, which is
// a quarter of one 128-element scale group, so exactly one scale load per
// chunk. Nibble decode: low = even element, high = odd; stored value is q+8.
constexpr int Q4_GROUP = 128;

__global__ void gemv_q4(const uint8_t* __restrict__ A, const __half* __restrict__ scales,
                        const __half* __restrict__ x, float* __restrict__ y, int M, int N) {
    const int lane = threadIdx.x;
    const int row  = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= M) return;
    const int groups = N / Q4_GROUP;
    const uint4*  a16  = reinterpret_cast<const uint4*>(A + (size_t)row * (N / 2)); // 32 nibbles
    const float4* x4   = reinterpret_cast<const float4*>(x);
    const __half* srow = scales + (size_t)row * groups;
    const int n32 = N / 32;
    float acc = 0.f;
    for (int i = lane; i < n32; i += 32) {
        const uint4 av = __ldg(&a16[i]);
        const float s = __half2float(__ldg(&srow[(i * 32) / Q4_GROUP]));
        float4 xv[4];
        #pragma unroll
        for (int k = 0; k < 4; ++k) xv[k] = __ldg(&x4[4 * i + k]);
        const uint8_t* ab = reinterpret_cast<const uint8_t*>(&av);
        const __half2* xh = reinterpret_cast<const __half2*>(xv); // 16 half2 = 32 elems
        float part = 0.f;
        #pragma unroll
        for (int b = 0; b < 16; ++b) {
            const float2 f = __half22float2(xh[b]);
            part = fmaf((float)((int)(ab[b] & 0xF) - 8), f.x, part);
            part = fmaf((float)((int)(ab[b] >> 4) - 8),  f.y, part);
        }
        acc = fmaf(part, s, acc); // whole chunk shares one group scale
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) y[row] = acc;
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

static double check(const std::vector<float>& got, const std::vector<double>& ref) {
    double worst = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const double rel = std::abs(got[i] - ref[i]) / (std::abs(ref[i]) + 1.0);
        worst = std::max(worst, rel);
    }
    return worst;
}

int main(int argc, char** argv) {
    const bool profile_mode = argc > 1 && std::string(argv[1]) == "--profile";
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("device: %s  (sm_%d%d, %d SMs, %.1f GB)\n",
                prop.name, prop.major, prop.minor, prop.multiProcessorCount,
                prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));

    auto read_ceiling = [&]() -> double {
        const size_t probe_bytes = 256ull << 20;
        const size_t n4 = probe_bytes / sizeof(float4);
        float4* psrc;
        float* psink;
        CUDA_CHECK(cudaMalloc(&psrc, probe_bytes));
        CUDA_CHECK(cudaMalloc(&psink, sizeof(float)));
        CUDA_CHECK(cudaMemset(psrc, 1, probe_bytes));
        const int pblocks = prop.multiProcessorCount * 8;
        Timing tr = time_kernel([&] { read_kernel<<<pblocks, 256>>>(psrc, psink, n4); });
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaFree(psrc));
        CUDA_CHECK(cudaFree(psink));
        return probe_bytes / (tr.ms * 1e-3) / 1e9;
    };
    double read_gbs = 1.0;
    if (!profile_mode) read_gbs = read_ceiling();

    struct Shape { int M, N; };
    const Shape shapes[] = {{4096, 4096}, {8192, 8192}, {14336, 4096}, {4096, 14336}};

    struct Row { int M, N; const char* name; float ms; double gbs, err; float speedup; };
    std::vector<Row> rows;

    for (const Shape s : shapes) {
        const int M = s.M, N = s.N;
        if (N % Q4_GROUP != 0) { std::fprintf(stderr, "N must be a multiple of %d\n", Q4_GROUP); return 1; }
        const int groups = N / Q4_GROUP;

        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(-1.f, 1.f);
        std::vector<float>  fA((size_t)M * N);
        std::vector<__half> hA((size_t)M * N), hx(N);
        for (size_t i = 0; i < fA.size(); ++i) { fA[i] = dist(rng); hA[i] = __float2half(fA[i]); }
        for (auto& v : hx) v = __float2half(dist(rng));

        // ---- host quantization ----
        // q8: symmetric per-row
        std::vector<int8_t> q8((size_t)M * N);
        std::vector<float>  s8(M);
        for (int i = 0; i < M; ++i) {
            float mx = 0.f;
            for (int j = 0; j < N; ++j) mx = std::max(mx, std::abs(fA[(size_t)i * N + j]));
            const float sc = mx > 0.f ? mx / 127.f : 1.f;
            s8[i] = sc;
            for (int j = 0; j < N; ++j) {
                const int q = (int)std::lround(fA[(size_t)i * N + j] / sc);
                q8[(size_t)i * N + j] = (int8_t)std::clamp(q, -127, 127);
            }
        }
        // q4: symmetric per group of 128, stored as (q+8) nibbles, low = even elem
        std::vector<uint8_t> q4((size_t)M * N / 2);
        std::vector<__half>  s4((size_t)M * groups);
        for (int i = 0; i < M; ++i) {
            for (int g = 0; g < groups; ++g) {
                float mx = 0.f;
                for (int j = g * Q4_GROUP; j < (g + 1) * Q4_GROUP; ++j)
                    mx = std::max(mx, std::abs(fA[(size_t)i * N + j]));
                const float sc = mx > 0.f ? mx / 7.f : 1.f;
                s4[(size_t)i * groups + g] = __float2half(sc);
                for (int j = g * Q4_GROUP; j < (g + 1) * Q4_GROUP; j += 2) {
                    const int q0 = std::clamp((int)std::lround(fA[(size_t)i * N + j]     / sc), -7, 7);
                    const int q1 = std::clamp((int)std::lround(fA[(size_t)i * N + j + 1] / sc), -7, 7);
                    q4[((size_t)i * N + j) / 2] = (uint8_t)((q0 + 8) | ((q1 + 8) << 4));
                }
            }
        }

        // ---- double-precision references (per weight format) + quant rmse ----
        auto ref_for = [&](auto&& wfun) {
            std::vector<double> r(M, 0.0);
            for (int i = 0; i < M; ++i) {
                double acc = 0.0;
                for (int j = 0; j < N; ++j)
                    acc += wfun(i, j) * (double)__half2float(hx[j]);
                r[i] = acc;
            }
            return r;
        };
        const std::vector<double> refF = ref_for([&](int i, int j) {
            return (double)__half2float(hA[(size_t)i * N + j]); });
        const std::vector<double> ref8 = ref_for([&](int i, int j) {
            return (double)q8[(size_t)i * N + j] * (double)s8[i]; });
        const std::vector<double> ref4 = ref_for([&](int i, int j) {
            const uint8_t b = q4[((size_t)i * N + j) / 2];
            const int q = (j % 2 == 0) ? (int)(b & 0xF) - 8 : (int)(b >> 4) - 8;
            return (double)q * (double)__half2float(s4[(size_t)i * groups + j / Q4_GROUP]); });

        double rmse8 = 0.0, rmse4 = 0.0;
        for (size_t i = 0; i < fA.size(); ++i) {
            const int row_i = (int)(i / N), col = (int)(i % N);
            const double d8 = (double)q8[i] * s8[row_i] - fA[i];
            const uint8_t b = q4[i / 2];
            const int q = (col % 2 == 0) ? (int)(b & 0xF) - 8 : (int)(b >> 4) - 8;
            const double d4 = (double)q * (double)__half2float(s4[(size_t)row_i * groups + col / Q4_GROUP]) - fA[i];
            rmse8 += d8 * d8; rmse4 += d4 * d4;
        }
        rmse8 = std::sqrt(rmse8 / (double)fA.size());
        rmse4 = std::sqrt(rmse4 / (double)fA.size());

        // ---- device buffers; weights rotate through >=256MB so every timed
        //      iteration streams cold from DRAM (Ada L2 is ~48MB) ----
        auto ncopies_for = [](size_t b) { return (int)std::max<size_t>(1, ((256ull << 20) + b - 1) / b); };
        const size_t bF = (size_t)M * N * 2, b8 = (size_t)M * N, b4 = (size_t)M * N / 2;
        const int cF = ncopies_for(bF), c8 = ncopies_for(b8), c4 = ncopies_for(b4);

        __half *dF, *dx, *ds4;
        int8_t* d8;
        uint8_t* d4;
        float *dy, *ds8;
        CUDA_CHECK(cudaMalloc(&dF, bF * cF));
        CUDA_CHECK(cudaMalloc(&d8, b8 * c8));
        CUDA_CHECK(cudaMalloc(&d4, b4 * c4));
        CUDA_CHECK(cudaMalloc(&dx, (size_t)N * 2));
        CUDA_CHECK(cudaMalloc(&dy, (size_t)M * 4));
        CUDA_CHECK(cudaMalloc(&ds8, (size_t)M * 4));
        CUDA_CHECK(cudaMalloc(&ds4, (size_t)M * groups * 2));
        for (int c = 0; c < cF; ++c) CUDA_CHECK(cudaMemcpy(dF + c * (size_t)M * N, hA.data(), bF, cudaMemcpyHostToDevice));
        for (int c = 0; c < c8; ++c) CUDA_CHECK(cudaMemcpy(d8 + c * (size_t)M * N, q8.data(), b8, cudaMemcpyHostToDevice));
        for (int c = 0; c < c4; ++c) CUDA_CHECK(cudaMemcpy(d4 + c * (size_t)M * N / 2, q4.data(), b4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dx, hx.data(), (size_t)N * 2, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(ds8, s8.data(), (size_t)M * 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(ds4, s4.data(), (size_t)M * groups * 2, cudaMemcpyHostToDevice));

        size_t itF = 0, it8 = 0, it4 = 0;
        std::vector<float> out(M);
        float fp16_ms = 0.f;

        auto bench = [&](const char* name, double bytes, const std::vector<double>& ref, auto&& launch) {
            launch();
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(out.data(), dy, M * 4, cudaMemcpyDeviceToHost));
            const double err = check(out, ref);
            if (err > 1e-2) {
                std::fprintf(stderr, "%s FAILED correctness: max rel err %.3g\n", name, err);
                std::exit(1);
            }
            if (profile_mode) {
                std::printf("%dx%d %s: correct (max rel err %.2e), launched once for ncu\n", M, N, name, err);
                return;
            }
            const Timing t = time_kernel(launch);
            if (std::string(name) == "fp16") fp16_ms = t.ms;
            rows.push_back({M, N, name, t.ms, bytes / (t.ms * 1e-3) / 1e9, err,
                            fp16_ms > 0.f ? fp16_ms / t.ms : 1.f});
        };

        const dim3 grid((M + 7) / 8), block(32, 8);
        bench("fp16", (double)M * N * 2 + N * 2 + M * 4, refF,
              [&] { gemv_vec<<<grid, block>>>(dF + (itF++ % cF) * (size_t)M * N, dx, dy, M, N); });
        bench("q8", (double)M * N * 1 + M * 4 + N * 2 + M * 4, ref8,
              [&] { gemv_q8<<<grid, block>>>(d8 + (it8++ % c8) * (size_t)M * N, ds8, dx, dy, M, N); });
        bench("q4", (double)M * N / 2 + (double)M * groups * 2 + N * 2 + M * 4, ref4,
              [&] { gemv_q4<<<grid, block>>>(d4 + (it4++ % c4) * (size_t)M * N / 2, ds4, dx, dy, M, N); });

        if (!profile_mode)
            std::printf("quantization rmse @ %dx%d:  q8 %.2e   q4 %.2e   (weights ~U[-1,1])\n",
                        M, N, rmse8, rmse4);
        else
            std::printf("\n");

        CUDA_CHECK(cudaFree(dF)); CUDA_CHECK(cudaFree(d8)); CUDA_CHECK(cudaFree(d4));
        CUDA_CHECK(cudaFree(dx)); CUDA_CHECK(cudaFree(dy));
        CUDA_CHECK(cudaFree(ds8)); CUDA_CHECK(cudaFree(ds4));

        // re-probe the ceiling after every shape: laptop clocks drift within a
        // suite, and the denominator must have seen the same best clock the
        // kernels did or MBU can read >100%
        if (!profile_mode) read_gbs = std::max(read_gbs, read_ceiling());
    }

    if (!profile_mode) {
        std::printf("\npure-read ceiling (max of before/after): %.1f GB/s  = MBU denominator\n\n", read_gbs);
        std::printf("%-12s %-6s %10s %10s %8s %9s   %s\n",
                    "shape", "kernel", "ms", "GB/s", "MBU%", "x fp16", "max-rel-err");
        long long prevShape = 0;
        for (const Row& r : rows) {
            const long long shape = (long long)r.M << 20 | r.N;
            if (prevShape != 0 && shape != prevShape) std::printf("\n");
            prevShape = shape;
            std::printf("%-12s %-6s %10.4f %10.1f %8.1f %8.2fx   %.2e\n",
                        (std::to_string(r.M) + "x" + std::to_string(r.N)).c_str(), r.name,
                        r.ms, r.gbs, 100.0 * r.gbs / read_gbs, r.speedup, r.err);
        }
    }

    std::printf("\ndone. all kernels passed the double-precision correctness check.\n");
    return 0;
}
