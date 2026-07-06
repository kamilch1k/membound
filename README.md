# membound

Hand-written **CUDA kernels for batch-1 LLM decode** — FP16 GEMV at **99.6%
of measured memory bandwidth**, online softmax at **98.1%**, and INT8/INT4
dequant GEMV turning compression ratio into **2×/3.9× tokens/s** — profiled
with Nsight Compute at every stage. Zero dependencies beyond the CUDA
toolkit; every benchmarked kernel is correctness-checked against a
double-precision reference on every run.

```
pure-read ceiling (max of before/after): 419.4 GB/s  = MBU denominator

shape        kernel          ms       GB/s     MBU%
8192x8192    naive       1.4724       91.2     21.7
8192x8192    warp        0.3256      412.4     98.3
8192x8192    vec         0.3215      417.6     99.6
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

| shape       | kernel | ms     | GB/s  | MBU%     | max rel err |
|-------------|--------|--------|-------|----------|-------------|
| 4096×4096   | naive  | 0.4460 | 75.3  | 17.9     | 3.1e-05     |
| 4096×4096   | warp   | 0.0840 | 399.7 | 95.3     | 9.0e-06     |
| 4096×4096   | vec    | 0.0821 | 409.0 | 97.5     | 1.0e-05     |
| 8192×8192   | naive  | 1.4724 | 91.2  | 21.7     | 8.1e-05     |
| 8192×8192   | warp   | 0.3256 | 412.4 | 98.3     | 2.5e-05     |
| 8192×8192   | vec    | 0.3215 | 417.6 | **99.6** | 1.9e-05     |
| 14336×4096  | naive  | 1.2030 | 97.7  | 23.3     | 4.4e-05     |
| 14336×4096  | warp   | 0.3136 | 374.7 | 89.3     | 1.2e-05     |
| 14336×4096  | vec    | 0.3105 | 378.4 | 90.2     | 1.0e-05     |
| 4096×14336  | naive  | 1.5900 | 73.9  | 17.6     | 1.3e-04     |
| 4096×14336  | warp   | 0.3068 | 382.9 | 91.3     | 2.6e-05     |
| 4096×14336  | vec    | 0.2815 | 417.3 | **99.5** | 2.3e-05     |

## The stages, and what Nsight says about each

Nsight Compute metrics for the 8192×8192 launches
(`artifacts/ncu-metrics.csv` has all four shapes):

| kernel | DRAM throughput (% of peak) | sectors / request | achieved occupancy | L2 hit % |
|--------|------|------|------|-----|
| naive  | 22.0 | 16.5 | 15.7 | 2.0 |
| warp   | 93.8 | 2.0  | 97.9 | 2.4 |
| vec    | 96.9 | 16.0 | 95.5 | 1.0 |

1. **naive** — one thread per output row. Adjacent threads read rows N elements
   apart, so one warp-level load touches 32 different cache lines. Nsight shows
   the crime directly: **16.5 sectors fetched per 32-thread request** that only
   uses 64 bytes — ~8× the traffic the kernel consumes — plus 16% occupancy
   because a 4096-row matrix only fills 16 blocks on a 58-SM chip.
2. **warp** — one warp per row: the 32 lanes read consecutive halves, and
   sectors/request drops to the ideal **2.0** (32 lanes × 2 B = 64 B = exactly
   2 sectors, all consumed). x is staged through a shared-memory tile; partial
   sums combine with a `__shfl_down_sync` reduction. DRAM throughput jumps
   22% → 94%.
3. **vec** — 128-bit loads: each lane reads a `float4` (8 halves) per step, so
   the same coalesced traffic moves in 8× fewer, full-width transactions
   (16 sectors per request — every byte used). FP32 accumulation via
   `__half22float2` + FMA. 96.9% DRAM throughput; the benchmark's best case is
   99.6% of the measured ceiling.

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
   clock — exceeded it ("104% MBU"). The harness now measures the pure-read
   ceiling before *and* after the suite and normalizes by the max, so the
   denominator reflects the same sustained clock the kernels enjoy.

## Bonus kernel: online softmax

`src/softmax.cu` — the other memory-bound decode kernel, same treatment.
A fused kernel computes **max and sum in a single sweep** with the online
update (`m' = max(m,x); d = d·e^(m−m') + e^(x−m')`, Milakov & Gimelshein),
then normalizes in a second pass that re-reads the row through L2. Floor
traffic is one read + one write per element — a balanced stream — so the MBU
denominator here is the measured **copy** ceiling, not pure-read (using GEMV's
read ceiling would flatter the number; the denominator has to match the
traffic mix).

| shape        | kernel | ms      | GB/s  | MBU%     |
|--------------|--------|---------|-------|----------|
| 8192×8192    | naive  | 6.7575  | 39.7  | 10.8     |
| 8192×8192    | online | 0.7433  | 361.1 | **98.1** |
| 4096×32768   | naive  | 17.6985 | 30.3  | 8.2      |
| 4096×32768   | online | 1.7674  | 303.8 | 82.6     |
| 1024×131072  | naive  | 70.8439 | 7.6   | 2.1      |
| 1024×131072  | online | 2.1622  | 248.3 | 67.5     |

The falloff at 131072-wide rows is not noise — it is L2 capacity, and it is
quantitatively predictable: 256 KB rows × ~232 resident blocks ≈ 59 MB > 48 MB
of L2, so pass 2's re-read spills to DRAM and the kernel pays ~3 passes
against a 2-pass floor → predicted 2/3 = 66.7% MBU, measured 67.5%. (Upgrade
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

| shape      | kernel | ms     | MBU% | × fp16    |
|------------|--------|--------|------|-----------|
| 8192×8192  | fp16   | 0.3215 | 99.5 | 1.00×     |
| 8192×8192  | q8     | 0.1749 | 91.5 | 1.84×     |
| 8192×8192  | q4     | 0.0916 | 90.1 | 3.51×     |
| 14336×4096 | fp16   | 0.3441 | 81.4 | 1.00×     |
| 14336×4096 | q8     | 0.1668 | 84.1 | **2.06×** |
| 14336×4096 | q4     | 0.0875 | 82.6 | **3.93×** |

The dequant kernels hold within a few points of the FP16 kernel's MBU, so the
speedup tracks the compression ratio: **~2× for INT8, ~3.5–3.9× for INT4**
(INT4's ratio is 3.76×, not 4× — the group scales cost 1/64th of the weight
bytes, and the harness counts them). The speedup column is also
clock-independent — each shape's three kernels run back-to-back at the same
power state, so it stays honest even on later shapes where laptop thermal
throttling drags every kernel's absolute MBU down together.

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
`s_memrealtime`-based probing on an MI300X. The kernels were written to be
retuned, not ported line-by-line: the benchmark harness and its methodology
are the transferable asset.

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
