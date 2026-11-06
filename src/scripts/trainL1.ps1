[CmdletBinding()]
param(
    [string]$Root = "artifacts/radar_ln3",
    [ValidateSet("cuda", "cpu")][string]$Device = "cuda",
    [int]$BatchSize = 4,
    [int]$ProgressInterval = 25,
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
$endpoint = Join-Path $PSScriptRoot "train_endpoint.ps1"
$forwardArguments = @{
    Root             = $Root
    Stage            = "l1"
    L2Mode           = "residual"
    Device           = $Device
    BatchSize        = $BatchSize
    ProgressInterval = $ProgressInterval
    NumWorkers       = $NumWorkers
    PrefetchFactor   = $PrefetchFactor
    PinMemory        = [bool]$PinMemory
    NoPinMemory      = [bool]$NoPinMemory
    Overwrite        = [bool]$Overwrite
    RebuildCache     = [bool]$RebuildCache
    SkipDataValidation = [bool]$SkipDataValidation
    ReplaceExisting  = [bool]$ReplaceExisting
}
if ($PythonPath) { $forwardArguments.PythonPath = $PythonPath }
& $endpoint @forwardArguments
if ($LASTEXITCODE -ne 0) { throw "L1 endpoint training failed, exit code=$LASTEXITCODE" }
