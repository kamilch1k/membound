# Elevated Nsight Compute run: profiles every gemv_* launch of `membound --profile`
# (each kernel launches exactly once per shape) and writes UTF-8 CSV + a fresh
# benchmark run into artifacts/. Launch with admin rights (GPU perf counters).
$ErrorActionPreference = "Continue"
$env:PYTHONNOUSERSITE = "1"
$repo = "C:\cc\membound"
$ncu  = "C:\Program Files\NVIDIA Corporation\Nsight Compute 2026.2.1\target\windows-desktop-win7-x64\ncu.exe"
New-Item -ItemType Directory -Force "$repo\artifacts" | Out-Null

$metrics = @(
    "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "lts__t_sector_hit_rate.pct",
    "l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio"
) -join ","

# cmd redirection keeps ncu's CSV bytes as-is (PowerShell > would re-encode UTF-16)
cmd /c "`"$ncu`" --kernel-name regex:gemv_.* --metrics $metrics --csv `"$repo\build\membound.exe`" --profile > `"$repo\artifacts\ncu-metrics.csv`" 2> `"$repo\artifacts\ncu-stderr.log`""
$ncuExit = $LASTEXITCODE

# fresh benchmark numbers in the same session (plug into AC first!)
cmd /c "`"$repo\build\membound.exe`" > `"$repo\artifacts\bench.txt`" 2>&1"
"ncu=$ncuExit bench=$LASTEXITCODE" | Out-File "$repo\artifacts\ncu-done.txt" -Encoding utf8
