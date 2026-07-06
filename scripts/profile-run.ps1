# Elevated Nsight Compute run: profiles every kernel launch of all three
# benchmark binaries in --profile mode (each kernel launches exactly once) and
# writes pure CSVs + fresh benchmark runs into artifacts/. Needs admin rights
# (GPU perf counters are admin-only on Windows unless unlocked in the driver).
$ErrorActionPreference = "Continue"
$env:PYTHONNOUSERSITE = "1"  # ncu bundles Python; a user site-packages here breaks its startup
$repo = Split-Path $PSScriptRoot -Parent

# ncu from PATH if available, else the default Nsight Compute install location
$ncu = (Get-Command ncu -ErrorAction SilentlyContinue).Source
if (-not $ncu) {
    $ncu = Get-ChildItem "C:\Program Files\NVIDIA Corporation\Nsight Compute*\target\windows-desktop-win7-x64\ncu.exe" |
           Sort-Object FullName -Descending | Select-Object -First 1 -ExpandProperty FullName
}
if (-not $ncu) { throw "ncu.exe not found — install Nsight Compute or add it to PATH" }

New-Item -ItemType Directory -Force "$repo\artifacts" | Out-Null

$metrics = @(
    "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "lts__t_sector_hit_rate.pct",
    "l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio"
) -join ","

$jobs = @(
    @{ exe = "membound.exe";         regex = "regex:gemv_.*";    csv = "ncu-metrics.csv"; bench = "bench.txt" },
    @{ exe = "membound-softmax.exe"; regex = "regex:softmax_.*"; csv = "ncu-softmax.csv"; bench = "bench-softmax.txt" },
    @{ exe = "membound-dequant.exe"; regex = "regex:gemv_.*";    csv = "ncu-dequant.csv"; bench = "bench-dequant.txt" }
)

foreach ($j in $jobs) {
    # ncu interleaves the app's stdout with its CSV; capture raw (via cmd so the
    # bytes aren't re-encoded UTF-16 by PowerShell), then split at the CSV header
    # so the committed .csv actually parses as CSV.
    $raw = Join-Path $env:TEMP "ncu-raw.txt"
    cmd /c "`"$ncu`" --kernel-name $($j.regex) --metrics $metrics --csv `"$repo\build\$($j.exe)`" --profile > `"$raw`" 2>nul"
    $lines = Get-Content $raw
    $hdr = ($lines | Select-String -SimpleMatch '"ID","Process ID"' | Select-Object -First 1).LineNumber
    if ($hdr) {
        $lines[($hdr - 1)..($lines.Count - 1)] | Set-Content "$repo\artifacts\$($j.csv)" -Encoding ascii
    } else {
        Write-Warning "$($j.exe): no CSV header in ncu output (counters not permitted?)"
    }
    cmd /c "`"$repo\build\$($j.exe)`" > `"$repo\artifacts\$($j.bench)`" 2>&1"
}
# completion sentinel outside the repo — artifacts/ holds only reviewed output
"done $(Get-Date -Format s)" | Out-File (Join-Path $env:TEMP "membound-profile-done.txt") -Encoding utf8
