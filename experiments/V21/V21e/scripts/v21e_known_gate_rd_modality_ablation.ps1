param(
    [string]$ProjectRoot = ''
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

try {
    $scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$srcRoot = Split-Path -Parent $scriptRoot
$defaultProjectRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $srcRoot))
$ResolvedRoot = (Resolve-Path -LiteralPath $(if ([string]::IsNullOrWhiteSpace($ProjectRoot)) { $defaultProjectRoot } else { $ProjectRoot })).Path
    $Python = Join-Path $ResolvedRoot '.venv\Scripts\python.exe'
    $SourceRoot = $srcRoot
    if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
        throw "Python executable not found: $Python"
    }
    Push-Location $SourceRoot
    try {
        & $Python -X utf8 -m report.known_gate_v21e_rd_modality_ablation --project-root $ResolvedRoot
        if ($LASTEXITCODE -ne 0) {
            throw "V21e RD modality ablation failed with exit code $LASTEXITCODE"
        }
    }
    finally {
        Pop-Location
    }
}
catch {
    Write-Error ('V21e KnownGate RD modality read-only ablation stopped: ' + $_.Exception.Message)
    exit 1
}
