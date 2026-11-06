[CmdletBinding()]
param(
    [ValidateSet("residual", "direct", "direct_grouped")][string]$L2Mode = "residual",
    [ValidateSet("cuda", "cpu")][string]$Device = "cuda",
    [ValidateRange(1, 64)][int]$BatchSize = 4,
    [ValidateRange(1, 10000)][int]$ProgressInterval = 25,
    [int]$NumWorkers = -1,
    [ValidateRange(1, 16)][int]$PrefetchFactor = 2,
    [switch]$PinMemory,
    [switch]$NoPinMemory,
    [switch]$RebuildCache,
    [switch]$SkipDataValidation,
    [switch]$ReplaceExisting,
    [switch]$Overwrite,
    [switch]$Resume,
    [string]$PythonPath = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
if ($Resume -and ($Overwrite -or $ReplaceExisting)) {
    throw "-Resume 不能与 -Overwrite 或 -ReplaceExisting 同时使用。"
}
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
if ([string]::IsNullOrWhiteSpace($PythonPath)) {
    $PythonPath = Join-Path $projectRoot ".venv\Scripts\python.exe"
}
$outputPath = "experiments/tmp/L2/l2/l2.pt"
$logRoot = Join-Path $PSScriptRoot "logs"
New-Item -ItemType Directory -Force -Path $logRoot | Out-Null
$logPath = Join-Path $logRoot ("trainL2-endpoint-{0}.log" -f [DateTime]::UtcNow.ToString("yyyyMMddTHHmmssfffZ"))
$endpoint = Join-Path $projectRoot "src\scripts\trainL2.ps1"
$forwardArguments = @{
    Root             = "experiments/tmp/L2"
    L2Mode           = $L2Mode
    Device           = $Device
    BatchSize        = $BatchSize
    ProgressInterval = $ProgressInterval
    NumWorkers       = $NumWorkers
    PrefetchFactor   = $PrefetchFactor
    PinMemory        = [bool]$PinMemory
    NoPinMemory      = [bool]$NoPinMemory
    PythonPath       = $PythonPath
}
if ($RebuildCache) { $forwardArguments.RebuildCache = $true }
if ($SkipDataValidation) { $forwardArguments.SkipDataValidation = $true }
if ($ReplaceExisting) { $forwardArguments.ReplaceExisting = $true }
if ($Overwrite) { $forwardArguments.Overwrite = $true }
if ($Resume) { $forwardArguments.Resume = $true }

Write-Host "[trainL2] 训练 L2 endpoint；输出=$outputPath mode=$L2Mode device=$Device batch=$BatchSize"
Write-Host "[trainL2] 配置=src/config/model.json；Root=experiments/tmp/L2"
Write-Host "[trainL2] 日志=$logPath"
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
    & $endpoint @forwardArguments 2>&1 | Tee-Object -FilePath $logPath
    if ($LASTEXITCODE -ne 0) { throw "L2 endpoint training failed, exit code=$LASTEXITCODE; see $logPath" }
}
finally {
    if ($null -ne $cudaLock) { $cudaLock.Dispose() }
}
