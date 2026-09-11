[CmdletBinding()]
param(
    [ValidateSet(
        'prepare',
        'smoke',
        'init-unfitted-l3',
        'prepare-open-set',
        'materialize-representations',
        'train-l1',
        'train-l2',
        'train-l1-knownness',
        'train-l1-quality',
        'train-l2-knownness',
        'train-l2-quality',
        'finalize-l1-heads',
        'finalize-l2-heads',
        'materialize-head-scores',
        'calibrate',
        'evaluate-open-set',
        'audit'
    )]
    [string]$Stage = 'smoke',
    [string]$ProjectRoot = ''
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$srcRoot = Split-Path -Parent $scriptRoot
$defaultRoot = Split-Path -Parent $srcRoot
$root = if ([string]::IsNullOrWhiteSpace($ProjectRoot)) { (Resolve-Path -LiteralPath $defaultRoot).Path } else { (Resolve-Path -LiteralPath $ProjectRoot).Path }
$python = Join-Path $root '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) { throw "Python interpreter is missing: $python" }

Push-Location -LiteralPath (Join-Path $root 'src-v3')
try {
    $arguments = @('-m', 'v27.v27_baseline.cli', '--project-root', $root)
    switch ($Stage) {
        'prepare' { $arguments += 'prepare' }
        'smoke' { $arguments += 'smoke' }
        'init-unfitted-l3' { $arguments += 'init-unfitted-l3' }
        'prepare-open-set' { $arguments += 'prepare-open-set' }
        'materialize-representations' { $arguments += 'materialize-representations' }
        'train-l1' { $arguments += @('train', '--task', 'known_gate') }
        'train-l2' { $arguments += @('train', '--task', 'dbb_specialist') }
        'train-l1-knownness' { $arguments += @('train-knownness', '--task', 'known_gate') }
        'train-l1-quality' { $arguments += @('train-quality', '--task', 'known_gate') }
        'train-l2-knownness' { $arguments += @('train-knownness', '--task', 'dbb_specialist') }
        'train-l2-quality' { $arguments += @('train-quality', '--task', 'dbb_specialist') }
        'finalize-l1-heads' { $arguments += @('finalize-heads', '--task', 'known_gate') }
        'finalize-l2-heads' { $arguments += @('finalize-heads', '--task', 'dbb_specialist') }
        'materialize-head-scores' { $arguments += 'materialize-head-scores' }
        'calibrate' { $arguments += 'calibrate' }
        'evaluate-open-set' { $arguments += 'evaluate-open-set' }
        'audit' { $arguments += 'audit' }
    }
    & $python @arguments
    if ($LASTEXITCODE -ne 0) { throw "V27 CLI failed with exit code $LASTEXITCODE" }
}
finally { Pop-Location }
