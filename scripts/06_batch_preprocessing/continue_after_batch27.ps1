param(
    [Parameter(Mandatory = $true)]
    [int]$BatchProcessId,
    [string]$WorkRoot = 'D:\ieum_local_pilot'
)
$ErrorActionPreference = 'Stop'
$lockPath = Join-Path $WorkRoot 'continue_after_batch27.lock'
$lock = [IO.File]::Open($lockPath, [IO.FileMode]::OpenOrCreate, [IO.FileAccess]::ReadWrite, [IO.FileShare]::None)
try {
    $batchProcess = Get-CimInstance Win32_Process -Filter "ProcessId=$BatchProcessId"
    if (-not $batchProcess -or $batchProcess.CommandLine -notmatch '14_run_local_batches.py' -or
        $batchProcess.CommandLine -notmatch 'BATCH_027') {
        throw 'Expected batch 27 process not found; refusing to start continuation.'
    }
    $started = $batchProcess.CreationDate
    Write-Output 'Waiting for batch 27 ZIP handoff; batch 27 will continue uninterrupted.'
    while ($true) {
        $current = Get-Process -Id $BatchProcessId -ErrorAction SilentlyContinue
        if (-not $current -or [math]::Abs(($current.StartTime - $started).TotalSeconds) -gt 1) { break }
        Start-Sleep -Seconds 2
    }
    $reports = @(Get-ChildItem -LiteralPath (Join-Path $WorkRoot 'batch_runs') -Filter 'BATCH_027_handoff_*.json' |
        Where-Object { $_.LastWriteTime -ge $started } | Sort-Object LastWriteTime -Descending)
    if ($reports.Count -eq 0) { throw 'No successful batch 27 ZIP handoff; continuation stopped.' }
    $report = Get-Content -Raw -Encoding UTF8 -LiteralPath $reports[0].FullName | ConvertFrom-Json
    if ($report.batch_id -ne 'BATCH_027' -or $report.status -ne 'LOCAL_HANDOFF_VERIFIED' -or
        -not (Test-Path -LiteralPath $report.archive)) {
        throw 'Batch 27 handoff is incomplete; continuation stopped.'
    }
    Write-Output 'Batch 27 ZIP handed off. Starting batch 26, then batch 25 without waiting for cloud sync.'
    & (Join-Path $PSScriptRoot 'run_local_batches.ps1') -BatchIds BATCH_026,BATCH_025 -Execute -RetryErrors -WorkRoot $WorkRoot
    exit $LASTEXITCODE
} finally {
    $lock.Dispose()
}
