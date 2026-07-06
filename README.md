# membound

Hand-written **FP16 GEMV CUDA kernels** pushed toward the memory-bandwidth
limit, profiled with Nsight Compute at every stage. Zero dependencies beyond
the CUDA toolkit.

<!-- TODO: headline MBU% number here once measured, matchbox-style -->

## Why GEMV, and why MBU

At batch size 1, LLM decode is a chain of matrix–vector products: every output
token streams each weight matrix from DRAM once and does ~1 FMA per weight.
Arithmetic is nearly free; the kernel is **memory-bandwidth-bound**. The metric
that counts is **MBU** — memory-bandwidth utilization: achieved GB/s divided by
the card's *measured* achievable bandwidth.

The datasheet number is not the ceiling — a laptop GPU is power- and
clock-limited. So the harness first measures the real ceiling with streaming
copy and pure-read probe kernels, and every MBU figure below is relative to
that measured pure-read ceiling (GEMV traffic is ~100% reads).

## Results

<!-- TODO: results table — stage | ms | GB/s | MBU% | Nsight DRAM% | sectors/req -->

Hardware: NVIDIA RTX 4080 Laptop (Ada, sm_89), CUDA 13.3, Windows 11.

## The stages

1. **naive** — one thread per output row. Adjacent threads read rows N elements
   apart, so each warp load touches 32 different cache lines: maximally
   uncoalesced. The "before".
2. **warp** — one warp per row: 32 lanes read consecutive elements (coalesced),
   x staged through a shared-memory tile and reused by every warp in the block,
   partial sums combined with a `__shfl_down_sync` reduction.
3. **vec** — 128-bit loads: each lane reads a `float4` (8 halves) per step,
   FP32 accumulation via `__half22float2` + FMA. Fewer, full-width transactions.

## Correctness

Every kernel is checked against a double-precision CPU reference **on every
run** before it is timed — max relative error is printed next to every
benchmark row. Inputs are seeded deterministically, so runs are reproducible.
No number in this README comes from an unverified kernel.

## Build & run

Requires the CUDA toolkit (13.x) and a C++20 host compiler.

```sh
cmake -G Ninja -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/membound
```

## Profiling

<!-- TODO: ncu commands used + per-stage metric table (DRAM throughput %,
     achieved occupancy, L2 hit rate, sectors/request) + artifacts/ dir -->
