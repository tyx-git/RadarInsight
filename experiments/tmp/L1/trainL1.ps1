[CmdletBinding()]
param(
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
    [string]$PythonPath = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
if ([string]::IsNullOrWhiteSpace($PythonPath)) {
    $PythonPath = Join-Path $projectRoot ".venv\Scripts\python.exe"
}
$outputPath = "experiments/tmp/L1/l1/l1.pt"
$logRoot = Join-Path $PSScriptRoot "logs"
New-Item -ItemType Directory -Force -Path $logRoot | Out-Null
$logPath = Join-Path $logRoot ("trainL1-endpoint-{0}.log" -f [DateTime]::UtcNow.ToString("yyyyMMddTHHmmssfffZ"))
$endpoint = Join-Path $projectRoot "src\scripts\trainL1.ps1"
$forwardArguments = @{
    Root             = "experiments/tmp/L1"
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

Write-Host "[trainL1] 训练 L1 endpoint；输出=$outputPath device=$Device batch=$BatchSize"
Write-Host "[trainL1] 配置=src/config/model.json；Root=experiments/tmp/L1"
Write-Host "[trainL1] 日志=$logPath"
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
    if ($LASTEXITCODE -ne 0) { throw "L1 endpoint training failed, exit code=$LASTEXITCODE; see $logPath" }
}
finally {
    if ($null -ne $cudaLock) { $cudaLock.Dispose() }
}
