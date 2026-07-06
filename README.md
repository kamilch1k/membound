# membound

[![ci](https://github.com/kamilch1k/membound/actions/workflows/ci.yml/badge.svg)](https://github.com/kamilch1k/membound/actions/workflows/ci.yml)

Hand-written **CUDA kernels for batch-1 LLM decode** — FP16 GEMV and online
softmax pushed to the measured memory-bandwidth limit, and INT8/INT4 dequant
GEMV turning compression ratio into **~2×/~3.9× GEMV throughput** (≈ tokens/s
in the GEMV-dominated decode limit) — profiled with Nsight Compute at every
stage. Zero dependencies beyond the CUDA toolkit; every benchmarked kernel is
correctness-checked against a double-precision reference on every run, and
warnings are errors on both host and device compilers.

```
MBU denominator: 417.0 GB/s  (pure-read probe 415.4, best-achieved 417.0)

shape        kernel          ms       GB/s     MBU%
8192x8192    naive       1.4230       94.4     22.6
8192x8192    warp        0.3325      403.8     96.8
8192x8192    vec         0.3284      408.8     98.0
4096x14336   vec         0.2817      417.0    100.0
```

## Why GEMV, and why MBU

At batch size 1, LLM decode is a chain of matrix–vector products: every output
token streams each weight matrix from DRAM once and does ~1 FMA per weight.
Arithmetic is nearly free; the kernel is **memory-bandwidth-bound**. The metric
that counts is **MBU** — memory-bandwidth utilization: achieved GB/s divided by
the card's *measured* achievable bandwidth.

Measured, not quoted from the datasheet: a laptop GPU is power- and
clock-limited, so the harness first establishes the real ceiling with streaming
copy and pure-read probe kernels (pure-read is the honest denominator for GEMV,
whose traffic is ~100% reads: `M·N·2` bytes read, `4·M` written).

## Results

RTX 4080 Laptop GPU (Ada, sm_89, 58 SMs), CUDA 13.3, Windows 11, plugged in.
`artifacts/` holds a complete regenerated run (benchmarks + Nsight CSVs);
absolute numbers move a few percent run-to-run with laptop power/thermal
state, which is why every MBU figure is normalized to a ceiling measured in
the same run. Shapes are typical decode projections (Llama-family
attention/MLP sizes).
100 timed iterations after 20 warmup; weights rotate through ≥256 MB of copies
so every iteration streams cold data from DRAM (see gotcha #1).

| shape       | kernel | ms     | GB/s  | MBU%      | max rel err |
|-------------|--------|--------|-------|-----------|-------------|
| 4096×4096   | naive  | 0.4480 | 74.9  | 18.0      | 3.1e-05     |
| 4096×4096   | warp   | 0.0840 | 399.8 | 95.9      | 9.0e-06     |
| 4096×4096   | vec    | 0.0893 | 376.1 | 90.2      | 1.0e-05     |
| 8192×8192   | naive  | 1.4230 | 94.4  | 22.6      | 8.1e-05     |
| 8192×8192   | warp   | 0.3325 | 403.8 | 96.8      | 2.5e-05     |
| 8192×8192   | vec    | 0.3284 | 408.8 | **98.0**  | 1.9e-05     |
| 14336×4096  | naive  | 1.1049 | 106.3 | 25.5      | 4.4e-05     |
| 14336×4096  | warp   | 0.2937 | 400.1 | 95.9      | 1.2e-05     |
| 14336×4096  | vec    | 0.2888 | 406.9 | 97.6      | 1.0e-05     |
| 4096×14336  | naive  | 1.5953 | 73.6  | 17.7      | 1.3e-04     |
| 4096×14336  | warp   | 0.3082 | 381.2 | 91.4      | 2.6e-05     |
| 4096×14336  | vec    | 0.2817 | 417.0 | **100.0** | 2.3e-05     |

(This table **is** `artifacts/bench.txt`, byte for byte — the README is
regenerated from the shipped artifacts, never the other way around.)

## The stages, and what Nsight says about each

Nsight Compute metrics for the 8192×8192 launches — the table is
`artifacts/ncu-metrics.csv`, which has all four shapes (captured with
`scripts/profile-run.ps1`; per-kernel counters like sectors/request are
structural and don't move run-to-run):

| kernel | DRAM throughput (% of peak) | sectors / request | achieved occupancy | L2 hit % |
|--------|-------|------|-------|-----|
| naive  | 22.82 | 16.5 | 15.59 | 1.9 |
| warp   | 93.80 | 2.0  | 97.93 | 2.4 |
| vec    | 97.19 | 16.0 | 95.27 | 1.0 |

1. **naive** — one thread per output row. Adjacent threads read rows N elements
   apart, so one warp-level load touches 32 different cache lines. Nsight shows
   the crime directly: **16.5 sectors fetched per 32-thread request** that only
   uses 64 bytes — ~8× the traffic the kernel consumes — plus 16% occupancy
   because a 4096-row matrix only fills 16 blocks on a 58-SM chip.
2. **warp** — one warp per row: the 32 lanes read consecutive halves, and
   sectors/request drops to the ideal **2.0** (32 lanes × 2 B = 64 B = exactly
   2 sectors, all consumed). x is staged through a shared-memory tile; partial
   sums combine with a `__shfl_down_sync` reduction. DRAM throughput jumps
   23% → 94%.
3. **vec** — 128-bit loads: each lane reads a `float4` (8 halves) per step, so
   the same coalesced traffic moves in 8× fewer, full-width transactions
   (16 sectors per request — every byte used). FP32 accumulation via
   `__half22float2` + FMA. The benchmark's best case is 417.0 GB/s —
   **at the measured limit**.

## Two gotchas the numbers caught

These are the reason the harness looks the way it does — both produced
impossible numbers first, and the fix is in the methodology:

1. **L2-resident weights.** Ada's L2 is ~48 MB; a 4096×4096 FP16 matrix is
   32 MB. Timed in a loop, iteration 2+ reads L2, not DRAM — the first run
   "achieved" 1.7 TB/s (424% MBU) on that shape. Real decode streams weights
   cold, so the harness cycles through ≥256 MB of weight copies; every
   iteration pays full DRAM cost. (x stays hot — realistic, activations are
   small.)
2. **Clock drift.** A laptop GPU's clocks move with power and thermal state.
   With the ceiling probed once at startup, kernels timed later — at a higher
   clock — exceeded it ("104% MBU"). The harness now re-probes the ceiling
   after every shape and takes the max; and because the probe and a kernel
   still sample different instants (and slightly different traffic mixes),
   the denominator also admits the best *achieved* kernel rate — an achieved
   rate is itself evidence of achievability. MBU ≤ 100 by construction, and
   100.0 reads "at the measured limit". Absolute numbers still move a few
   percent run-to-run with thermal state; the correctness columns don't
   (deterministic seed), and ratio columns are measured in interleaved rounds
   so they don't either.

## Bonus kernel: online softmax

`src/softmax.cu` — the other memory-bound decode kernel, same treatment.
A fused kernel computes **max and sum in a single sweep** with the online
update (`m' = max(m,x); d = d·e^(m−m') + e^(x−m')`, Milakov & Gimelshein),
then normalizes in a second pass that re-reads the row through L2. Floor
traffic is one read + one write per element — a balanced stream — so the MBU
denominator here is the measured **copy** ceiling, not pure-read (using GEMV's
read ceiling would flatter the number; the denominator has to match the
traffic mix).

| shape        | kernel | ms      | GB/s  | MBU%      |
|--------------|--------|---------|-------|-----------|
| 8192×8192    | naive  | 6.7400  | 39.8  | 10.3      |
| 8192×8192    | online | 0.6954  | 386.0 | **100.0** |
| 4096×32768   | naive  | 18.9618 | 28.3  | 7.3       |
| 4096×32768   | online | 1.7770  | 302.1 | 78.3      |
| 1024×131072  | naive  | 72.1288 | 7.4   | 1.9       |
| 1024×131072  | online | 2.0663  | 259.8 | 67.3      |

The falloff at 131072-wide rows is not noise — it is L2 capacity, and it is
quantitatively predictable: 256 KB rows × ~232 resident blocks ≈ 59 MB > 48 MB
of L2, so pass 2's re-read spills to DRAM and the kernel pays ~3 passes
against a 2-pass floor → predicted 2/3 = 66.7% MBU, measured 67.3%. (Upgrade
path if vocab-width rows mattered: split each row across blocks so the
per-chunk working set fits L2, at the cost of a two-level reduction.)

Also caught by the correctness gate: FP16 softmax outputs below ~6e-5 are
subnormal, where relative precision collapses by construction — the error
metric scores relative to `max(ref, 1e-4)` so real errors fail loudly while
subnormal rounding noise doesn't produce false alarms.

## The kernel decode actually runs: dequant GEMV

`src/dequant.cu` — once GEMV sits at the bandwidth limit, the only way to get
more tokens/s is to move fewer bytes. Quantized weights convert compression
ratio directly into decode speed — *if* the dequant kernel holds the same MBU
as the FP16 one. This benchmark tests exactly that claim: INT8 (symmetric
per-row scale) and INT4 (symmetric per-group-of-128 scales, AWQ/GGUF-style
granularity) weights streamed from DRAM, dequantized in-register, FP32
accumulate, with the FP16 `vec` kernel in the same binary as the baseline.

| shape      | kernel | ms     | MBU%  | × fp16    |
|------------|--------|--------|-------|-----------|
| 4096×4096  | fp16   | 0.0821 | 97.4  | 1.00×     |
| 4096×4096  | q8     | 0.0426 | 94.0  | 1.93×     |
| 4096×4096  | q4     | 0.0228 | 90.5  | 3.60×     |
| 8192×8192  | fp16   | 0.3213 | 99.5  | 1.00×     |
| 8192×8192  | q8     | 0.1622 | 98.7  | **1.98×** |
| 8192×8192  | q4     | 0.0825 | 100.0 | **3.89×** |
| 14336×4096 | fp16   | 0.2821 | 99.2  | 1.00×     |
| 14336×4096 | q8     | 0.1424 | 98.4  | 1.98×     |
| 14336×4096 | q4     | 0.0728 | 99.3  | 3.87×     |
| 4096×14336 | fp16   | 0.2809 | 99.6  | 1.00×     |
| 4096×14336 | q8     | 0.1422 | 98.4  | 1.97×     |
| 4096×14336 | q4     | 0.0734 | 98.4  | 3.83×     |

The dequant kernels hold within a few points of the FP16 kernel's MBU, so the
speedup tracks the compression ratio: **~2× for INT8, ~3.6–3.9× for INT4**.
INT4's theoretical cap is 3.87×, not 4×: the FP16 group scales add 2/128 B
per weight (1/32 of the INT4 weight bytes) for an effective 0.516 B/elem, and
the harness counts them. Measured ratios carry ~1% timing noise around that
cap. The three kernels are timed in **interleaved rounds** (best round per
kernel), because timing them in separate blocks lets laptop clock drift land
on one kernel and not another — measured before the fix as a "5.9×" speedup,
comfortably above what physics allows, which is exactly the kind of number
this harness exists to refuse.

Correctness here is checked against a double-precision reference computed on
the *dequantized* weights (`q·s` exactly), so it verifies the kernel's
arithmetic and indexing; quantization error itself is a modeling choice, and
is reported separately (`quant rmse`: 2.3e-3 for q8, 4.1e-2 for q4 on
U[−1,1] weights).

## A note on the AMD side

These kernels are CUDA on an Ada laptop GPU, because that is the hardware on
my desk. Nothing in the method is NVIDIA-specific: the levers (coalesced
sector-efficient loads, wide vector transactions, occupancy vs. register
pressure, measured-not-quoted ceilings, MBU as the metric) map directly onto
CDNA — 64-lane wavefronts, `dwordx4` loads, LDS staging, and
`s_memrealtime`-based probing on an MI300X. Some of the gotchas transform
rather than transfer: the L2-residency trap that faked 1.7 TB/s here becomes
a *bigger* trap on MI300X, whose 256 MB Infinity Cache can hold entire layers
of weights — the rotation working set has to scale accordingly, and the
softmax L2-spill cliff moves to much wider rows. The kernels were written to
be retuned, not ported line-by-line: the benchmark harness and its
methodology are the transferable asset.

## Correctness

Every kernel is checked against a double-precision CPU reference **on every
run** before it is timed — max relative error prints next to every benchmark
row, and the run aborts on failure. Inputs are seeded deterministically, so
runs are reproducible. No number in this README comes from an unverified
kernel.

## Build, run, profile

Requires the CUDA toolkit (13.x) and a C++20 host compiler.

```sh
cmake -G Ninja -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build        # correctness smoke tests (every kernel, GPU required)
./build/membound              # GEMV benchmark + correctness
./build/membound --profile    # each kernel launches exactly once, for ncu
./build/membound-softmax      # softmax benchmark + correctness
./build/membound-dequant      # INT8/INT4 dequant GEMV vs fp16 baseline
```

Profiling (GPU perf counters need admin on Windows, or the driver toggle):

```sh
ncu --kernel-name "regex:gemv_.*" \
    --metrics gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active,lts__t_sector_hit_rate.pct,l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio \
    --csv ./build/membound --profile
```

`scripts/profile-run.ps1` wraps this for Windows (elevated) and drops the CSV
into `artifacts/`.
