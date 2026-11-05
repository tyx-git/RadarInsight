[CmdletBinding()]
param(
    [string]$Root = "artifacts/radar_ln3",
    [ValidateSet("cuda", "cpu")][string]$Device = "cuda",
    [int]$BatchSize = 4,
    [int]$ProgressInterval = 25,
    [switch]$RebuildCache,
    [switch]$SkipDataValidation,
    [switch]$ReplaceExisting,
    [switch]$Overwrite,
    [switch]$Resume,
    [string]$PythonPath = ""
)
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$endpoint = Join-Path $PSScriptRoot "train_endpoint.ps1"
$args = @("-Root",$Root,"-Stage","l2","-L2Mode","residual","-Device",$Device,"-BatchSize",[string]$BatchSize,"-ProgressInterval",[string]$ProgressInterval)
if ($RebuildCache) {$args += "-RebuildCache"}
if ($SkipDataValidation) {$args += "-SkipDataValidation"}
if ($ReplaceExisting) {$args += "-ReplaceExisting"}
if ($Overwrite) {$args += "-Overwrite"}
if ($Resume) {$args += "-Resume"}
if ($PythonPath) {$args += @("-PythonPath",$PythonPath)}
& $endpoint @args
if ($LASTEXITCODE -ne 0) { throw "L2 endpoint training failed, exit code=$LASTEXITCODE" }
