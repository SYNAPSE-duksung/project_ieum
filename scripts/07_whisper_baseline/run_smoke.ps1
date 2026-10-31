param(
    [string]$DataRoot = 'H:\.shortcut-targets-by-id\15z9OYaGiLG_KO6Fuatrkgs1DNbD-D50q\ieum_project\00_data',
    [string]$Selection,
    [string]$Output,
    [int]$StopAfter = 0,
    [switch]$RetryFailures
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$pythonPath = Join-Path $projectRoot '.venv_whisper_baseline\Scripts\python.exe'
if (-not $Selection) { $Selection = Join-Path $projectRoot 'results\whisper_baseline\smoke_selection_20261027.json' }
if (-not $Output) { $Output = Join-Path $projectRoot 'results\whisper_baseline\smoke_20261027_v2' }
$env:PYTHONIOENCODING = 'utf-8'
$env:HF_HUB_DISABLE_XET = '1'
$runArgs = @('-B', (Join-Path $PSScriptRoot 'run_baseline.py'), '--data-root', $DataRoot, '--selection', $Selection, '--output', $Output)
if ($StopAfter -gt 0) { $runArgs += @('--stop-after', "$StopAfter") }
if ($RetryFailures) { $runArgs += '--retry-failures' }
& $pythonPath @runArgs
exit $LASTEXITCODE
