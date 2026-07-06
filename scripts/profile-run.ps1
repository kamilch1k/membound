# Elevated Nsight Compute run: profiles every kernel launch of all three
# benchmark binaries in --profile mode (each kernel launches exactly once) and
# writes UTF-8 CSVs + fresh benchmark runs into artifacts/. Launch with admin
# rights (GPU perf counters).
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

$jobs = @(
    @{ exe = "membound.exe";         regex = "regex:gemv_.*";    csv = "ncu-metrics.csv";         bench = "bench.txt" },
    @{ exe = "membound-softmax.exe"; regex = "regex:softmax_.*"; csv = "ncu-softmax.csv";         bench = "bench-softmax.txt" },
    @{ exe = "membound-dequant.exe"; regex = "regex:gemv_.*";    csv = "ncu-dequant.csv";         bench = "bench-dequant.txt" }
)

foreach ($j in $jobs) {
    # cmd redirection keeps ncu's CSV bytes as-is (PowerShell > would re-encode UTF-16)
    cmd /c "`"$ncu`" --kernel-name $($j.regex) --metrics $metrics --csv `"$repo\build\$($j.exe)`" --profile > `"$repo\artifacts\$($j.csv)`" 2> `"$repo\artifacts\ncu-stderr.log`""
    cmd /c "`"$repo\build\$($j.exe)`" > `"$repo\artifacts\$($j.bench)`" 2>&1"
}
"all done exit=$LASTEXITCODE" | Out-File "$repo\artifacts\ncu-done.txt" -Encoding utf8
