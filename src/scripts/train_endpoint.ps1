[CmdletBinding()]
param(
    [ValidateSet("all", "cache", "l1", "l2")][string]$Stage = "all",
    [ValidateSet("residual", "direct", "direct_grouped")][string]$L2Mode = "residual",
    [ValidateSet("cuda", "cpu")][string]$Device = "cuda",
    [switch]$RebuildCache,
    [switch]$SkipDataValidation,
    [switch]$SkipEventCacheValidation,
    [switch]$ReplaceExisting,
    [switch]$Resume,
    [ValidateRange(1, 10000)][int]$ProgressInterval = 5,
    [ValidateRange(1, 64)][int]$BatchSize = 4,
    [int]$NumWorkers = -1,
    [ValidateRange(1, 16)][int]$PrefetchFactor = 2,
    [switch]$PinMemory,
    [switch]$NoPinMemory,
    [string]$ModelConfigPath = "src/config/model.json",
    [string]$Root = "artifacts/radar_ln3",
    [switch]$Overwrite,
    [string]$PythonPath = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
if ($Resume -and ($ReplaceExisting -or $Overwrite)) {
    throw "-Resume 不能与 -ReplaceExisting 或 -Overwrite 同时使用；覆盖会清除可恢复的 epoch checkpoints。"
}
if ($NumWorkers -lt -1 -or $NumWorkers -gt 16) { throw "-NumWorkers 必须是 -1 或 0..16。" }
if ($PinMemory -and $NoPinMemory) { throw "-PinMemory 与 -NoPinMemory 不能同时使用。" }

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
if ([string]::IsNullOrWhiteSpace($PythonPath)) {
    $PythonPath = Join-Path $projectRoot ".venv\Scripts\python.exe"
} else {
    $PythonPath = (Resolve-Path -LiteralPath $PythonPath).Path
}
if (-not (Test-Path -LiteralPath $PythonPath -PathType Leaf)) {
    throw "Python 解释器不存在：$PythonPath"
}
$scriptPath = Join-Path $PSScriptRoot "train_endpoint.py"
$manifestPath = Join-Path $projectRoot "src\config\manifests.json"
if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) {
    throw "总 manifest 不存在：$manifestPath；请先完成原始数据处理和索引构建。"
}

$arguments = @(
    "-B", $scriptPath,
    "--project-root", $projectRoot,
    "--manifest-path", $manifestPath,
    "--stage", $Stage,
    "--l2-mode", $L2Mode,
    "--device", $Device,
    "--progress-interval", [string]$ProgressInterval,
    "--batch-size", [string]$BatchSize,
    "--prefetch-factor", [string]$PrefetchFactor,
    "--model-config-path", $ModelConfigPath,
    "--root", $Root
)
if ($RebuildCache) { $arguments += "--rebuild-cache" }
if ($SkipDataValidation) { $arguments += "--skip-data-validation" }
if ($SkipEventCacheValidation) { $arguments += "--skip-event-cache-validation" }
if ($ReplaceExisting) { $arguments += "--replace-existing" }
if ($Resume) { $arguments += "--resume" }
if ($Overwrite) { $arguments += "--overwrite" }
if ($NumWorkers -ge 0) { $arguments += @("--num-workers", [string]$NumWorkers) }
if ($PinMemory) { $arguments += "--pin-memory" }
if ($NoPinMemory) { $arguments += "--no-pin-memory" }

Write-Host "[train_endpoint] stage=$Stage root=$Root l2_mode=$L2Mode device=$Device batch_size=$BatchSize"
 $previousPythonPath = $env:PYTHONPATH
 $sourceRoot = Join-Path $projectRoot "src"
 $env:PYTHONPATH = if ([string]::IsNullOrWhiteSpace($previousPythonPath)) {
    $sourceRoot
} else {
    "$sourceRoot$([IO.Path]::PathSeparator)$previousPythonPath"
}
try {
    & $PythonPath @arguments
    $exitCode = $LASTEXITCODE
} finally {
    $env:PYTHONPATH = $previousPythonPath
}
if ($exitCode -ne 0) {
    throw "Endpoint training failed, Python exit code=$exitCode."
}
