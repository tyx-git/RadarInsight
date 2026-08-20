param(
    [string]$ProjectRoot = 'F:\Project\RadarInsight'
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

try {
    $ResolvedRoot = (Resolve-Path -LiteralPath $ProjectRoot).Path
    $Python = Join-Path $ResolvedRoot '.venv\Scripts\python.exe'
    $SourceRoot = Join-Path $ResolvedRoot 'src-v3'
    if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
        throw "Python executable not found: $Python"
    }
    Push-Location $SourceRoot
    try {
        & $Python -X utf8 -m smoke.known_gate_v21e_rd_modality_ablation_test --project-root $ResolvedRoot
        if ($LASTEXITCODE -ne 0) {
            throw "V21e RD modality ablation smoke failed with exit code $LASTEXITCODE"
        }
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
