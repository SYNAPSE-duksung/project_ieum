param(
    [switch]$Execute,
    [switch]$RetryErrors,
    [string[]]$BatchIds = @('BATCH_010'),
    [string]$UploadToken,
    [string]$WorkRoot = 'D:\ieum_local_pilot',
    [string]$StagingDir,
    [string]$OutputDir
)
$ErrorActionPreference = 'Stop'
$dataRoot = 'H:\.shortcut-targets-by-id\15z9OYaGiLG_KO6Fuatrkgs1DNbD-D50q\ieum_project\00_data'
$runnerPath = Join-Path $PSScriptRoot '14_run_local_batches.py'
$runnerArgs = @(
    $runnerPath,
    '--input-root', (Join-Path $dataRoot 'mini_dataset'),
    '--source-layout', 'drive-path',
    '--processed-dir', (Join-Path $WorkRoot 'processed_candidate2'),
    '--archive-dir', (Join-Path $WorkRoot 'archives'),
    '--sync-dir', (Join-Path $dataRoot 'processed_candidate2_archives'),
    '--candidate', (Join-Path $dataRoot 'batch_manifests\candidate_exclude_all_code_overlap_resolved.csv'),
    '--plan', (Join-Path $dataRoot 'batch_manifests\candidate2_batch_plan.csv')
)
if ($Execute) { $runnerArgs += '--execute' } else { $runnerArgs += '--dry-run' }
if ($RetryErrors) { $runnerArgs += '--retry-errors' }
if ($StagingDir) { $runnerArgs += '--staging-dir'; $runnerArgs += $StagingDir }
if ($BatchIds) { $runnerArgs += '--batch-ids'; $runnerArgs += $BatchIds }
if ($BatchIds.Count -eq 0) { throw 'Specify at least one batch.' }
if ($UploadToken) { $runnerArgs += '--upload-token'; $runnerArgs += $UploadToken }
if (-not $OutputDir) { $OutputDir = Join-Path $WorkRoot 'batch_runs' }
if ($OutputDir) { $runnerArgs += '--output-dir'; $runnerArgs += $OutputDir }
$pythonPath = Join-Path (Split-Path (Split-Path $PSScriptRoot -Parent) -Parent) '.venv_local_pilot\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) { $pythonPath = 'python' }
$env:Path = [Environment]::GetEnvironmentVariable('Path','User') + ';' + $env:Path
# Native stderr warnings must not abort Python on Windows PowerShell.
$ErrorActionPreference = 'Continue'
& $pythonPath @runnerArgs
exit $LASTEXITCODE
