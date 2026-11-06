[CmdletBinding()]
param(
    [ValidateSet("l1", "l2")][string]$Task = "l1",
    [ValidateSet("cuda", "cpu")][string]$Device = "cuda",
    [ValidateRange(1, 64)][int]$BatchSize = 4,
    [ValidateRange(1, 256)][int]$HeadBatchSize = 32,
    [ValidateRange(1, 10000)][int]$Epochs = 30,
    [double]$UnknownLossWeight = 0.15,
    [switch]$Overwrite,
    [switch]$SkipDataValidation,
    [string]$PythonPath = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
if ([string]::IsNullOrWhiteSpace($PythonPath)) {
    $PythonPath = Join-Path $projectRoot ".venv\Scripts\python.exe"
}
if (-not (Test-Path -LiteralPath $PythonPath -PathType Leaf)) {
    throw "Python interpreter does not exist: $PythonPath"
}
$headRoot = "experiments/tmp/L3/heads"
$representationRoot = "experiments/tmp/L3/representations"
$inputPaths = @(
    "experiments/tmp/L1/l1/l1.pt",
    "experiments/tmp/L2/l2/l2.pt",
    "data/cq08/cache/features/l3/head_fit/feature-cache-manifest.json",
    "data/cq08/cache/features/l3/validation/feature-cache-manifest.json"
)
foreach ($relativePath in $inputPaths) {
    if (-not (Test-Path -LiteralPath (Join-Path $projectRoot $relativePath) -PathType Leaf)) {
        throw "L3 knownness training input is missing: $relativePath"
    }
}
$headPath = if ($Task -eq "l1") { "$headRoot/l1-knownness.pt" } else { "$headRoot/l2-knownness.pt" }
$summaryPath = Join-Path $projectRoot ($headPath -replace '\.pt$', '-summary.json')
if (-not $Overwrite -and (Test-Path -LiteralPath (Join-Path $projectRoot $headPath) -PathType Leaf)) {
    throw "Knownness checkpoint already exists: $headPath; pass -Overwrite to replace it."
}
if (-not $Overwrite -and (Test-Path -LiteralPath $summaryPath -PathType Leaf)) {
    throw "Knownness summary already exists: $summaryPath; pass -Overwrite to replace it."
}
$logRoot = Join-Path $PSScriptRoot "logs"
New-Item -ItemType Directory -Force -Path $logRoot | Out-Null
$logPath = Join-Path $logRoot ("trainL3-{0}-knownness-{1}.log" -f $Task, [DateTime]::UtcNow.ToString("yyyyMMddTHHmmssfffZ"))

$arguments = @(
    "-B", "src/scripts/train_l3_heads.py",
    "--project-root", $projectRoot,
    "--manifest-path", "src/config/manifests.json",
    "--model-config-path", "src/config/model.json",
    "--l1-config-path", "src/config/model.json",
    "--l2-config-path", "src/config/model.json",
    "--l1-checkpoint-path", "experiments/tmp/L1/l1/l1.pt",
    "--l2-checkpoint-path", "experiments/tmp/L2/l2/l2.pt",
    "--fit-feature-cache-manifest-path", "data/cq08/cache/features/l3/head_fit/feature-cache-manifest.json",
    "--validation-feature-cache-manifest-path", "data/cq08/cache/features/l3/validation/feature-cache-manifest.json",
    "--representation-root", $representationRoot,
    "--output-root", $headRoot,
    "--task", $Task,
    "--device", $Device,
    "--batch-size", [string]$BatchSize,
    "--head-batch-size", [string]$HeadBatchSize,
    "--epochs", [string]$Epochs,
    "--skip-register"
)
if ($Task -eq "l1") {
    $arguments += @(
        "--l1-output-path", "$headRoot/l1-knownness.pt",
        "--unknown-loss-weight", [string]$UnknownLossWeight
    )
} else {
    $arguments += @("--l2-output-path", "$headRoot/l2-knownness.pt")
}
$l3Config = Get-Content -LiteralPath (Join-Path $projectRoot "src/config/radar_ln3_l3.json") -Raw -Encoding UTF8 | ConvertFrom-Json
if ($Overwrite) { $arguments += "--overwrite" }
if ($SkipDataValidation) {
    $arguments += @("--skip-data-validation", "--skip-event-cache-validation")
}

Write-Host "[trainL3] 训练 L3 $Task knownness head；L1/L2 endpoints 来自 experiments/tmp/L1 和 experiments/tmp/L2"
Write-Host "[trainL3] 输出=$headRoot device=$Device batch=$BatchSize head_batch=$HeadBatchSize epochs=$Epochs"
if ($Task -eq "l1") {
    if ($l3Config.knownness_training.l1.route_loss) {
        Write-Host "[trainL3] L1 route unknown_loss_weight=$UnknownLossWeight"
    } else {
        Write-Host "[trainL3] 注意：配置使用 balanced BCE，unknown_loss_weight=$UnknownLossWeight 不参与梯度；实际 Unknown BCE 权重=$($l3Config.knownness_training.l1.bce_unknown_weight)"
    }
}
Write-Host "[trainL3] 日志=$logPath"
$cudaLock = $null
if ($Device -eq "cuda") {
    $cudaLockPath = Join-Path $projectRoot "experiments\tmp\training-cuda.lock"
    try {
        $cudaLock = [System.IO.File]::Open(
            $cudaLockPath,
            [System.IO.FileMode]::OpenOrCreate,
            [System.IO.FileAccess]::ReadWrite,
            [System.IO.FileShare]::None
        )
    }
    catch [System.IO.IOException] {
        throw "Another experiments/tmp L1/L2/L3 CUDA training run is active. Wait for it to finish, then rerun this command."
    }
}
try {
    & $PythonPath @arguments 2>&1 | Tee-Object -FilePath $logPath
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) { throw "L3 $Task-knownness training failed, exit code=$exitCode; see $logPath" }
}
finally {
    if ($null -ne $cudaLock) { $cudaLock.Dispose() }
}
