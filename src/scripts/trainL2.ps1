[CmdletBinding()]
param(
    [string]$Root = "artifacts/radar_ln3",
    [ValidateSet("residual", "direct", "direct_grouped")][string]$L2Mode = "residual",
    [ValidateSet("cuda", "cpu")][string]$Device = "cuda",
    [int]$BatchSize = 4,
    [int]$ProgressInterval = 25,
    [int]$NumWorkers = -1,
    [ValidateRange(1, 16)][int]$PrefetchFactor = 2,
    [switch]$PinMemory,
    [switch]$NoPinMemory,
    [switch]$RebuildCache,
    [switch]$SkipDataValidation,
    [switch]$SkipEventCacheValidation,
    [switch]$ReplaceExisting,
    [switch]$Overwrite,
    [switch]$Resume,
    [string]$ModelConfigPath = "src/config/model.json",
    [string]$PythonPath = ""
)
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$endpoint = Join-Path $PSScriptRoot "train_endpoint.ps1"
$forwardArguments = @{
    Root               = $Root
    Stage              = "l2"
    L2Mode             = $L2Mode
    Device             = $Device
    BatchSize          = $BatchSize
    ProgressInterval   = $ProgressInterval
    NumWorkers         = $NumWorkers
    PrefetchFactor     = $PrefetchFactor
    PinMemory          = [bool]$PinMemory
    NoPinMemory        = [bool]$NoPinMemory
    ModelConfigPath    = $ModelConfigPath
    RebuildCache       = [bool]$RebuildCache
    SkipDataValidation = [bool]$SkipDataValidation
    SkipEventCacheValidation = [bool]$SkipEventCacheValidation
    ReplaceExisting    = [bool]$ReplaceExisting
    Overwrite          = [bool]$Overwrite
    Resume             = [bool]$Resume
}
if ($PythonPath) { $forwardArguments.PythonPath = $PythonPath }
Write-Host "[trainL2] endpoint training: mode=$L2Mode root=$Root device=$Device batch_size=$BatchSize"
& $endpoint @forwardArguments
if ($LASTEXITCODE -ne 0) { throw "L2 endpoint training failed, exit code=$LASTEXITCODE" }
