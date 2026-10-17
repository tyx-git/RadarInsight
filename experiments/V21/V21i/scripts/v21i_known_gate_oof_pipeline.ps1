[CmdletBinding()]
param(
    [ValidateSet('prepare', 'treatment', 'evaluate', 'all')]
    [string]$Stage = 'all',
    [ValidateRange(0, 4)]
    [int[]]$Folds = @(0, 1, 2, 3, 4),
    [string]$ProjectRoot = ''
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$generic = Join-Path $scriptRoot 'v21hi_known_gate_oof_pipeline.ps1'
& $generic -Variant 'i' -Stage $Stage -Folds $Folds -ProjectRoot $ProjectRoot
exit $LASTEXITCODE
