# Nsight Compute profiling for every kernel stage (run from repo root, needs ncu on PATH).
# Captures the metrics the README reports: DRAM throughput %, achieved occupancy,
# L2 hit rate, and sectors/request (coalescing quality).
$metrics = @(
    "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__warps_active.avg.pct_of_peak_sustained_active",      # achieved occupancy
    "lts__t_sector_hit_rate.pct",                             # L2 hit rate
    "l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio"
) -join ","

ncu --kernel-name "regex:gemv_.*" --launch-count 1 --launch-skip 20 `
    --metrics $metrics --csv --page raw `
    .\build\membound.exe | Tee-Object artifacts\ncu-metrics.csv
