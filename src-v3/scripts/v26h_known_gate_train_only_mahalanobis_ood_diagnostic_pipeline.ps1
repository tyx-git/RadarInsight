[CmdletBinding()]
param(
    [ValidateSet('prepare', 'smoke', 'diagnose', 'all')]
    [string]$Stage = 'all',
    [string]$ProjectRoot = ''
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$utf8NoBom = [System.Text.UTF8Encoding]::new($false)
[Console]::InputEncoding = $utf8NoBom
[Console]::OutputEncoding = $utf8NoBom
$OutputEncoding = $utf8NoBom

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$srcRoot = (Resolve-Path -LiteralPath (Split-Path -Parent $scriptRoot)).Path
$defaultProjectRoot = (Resolve-Path -LiteralPath (Split-Path -Parent $srcRoot)).Path
$projectPath = [System.IO.Path]::GetFullPath($(if ([string]::IsNullOrWhiteSpace($ProjectRoot)) { $defaultProjectRoot } else { $ProjectRoot }))
$python = Join-Path $projectPath '.venv\Scripts\python.exe'
$contractPath = Join-Path $projectPath 'data\processed\RADAR-LN3-V26h-cq08-l1-train-only-mahalanobis-ood-diagnostic-contract.json'
$readinessPath = Join-Path $projectPath 'src-v3\processed\report\RADAR-LN3-V26h-l1-train-only-mahalanobis-ood-diagnostic-readiness.json'
$reportPath = Join-Path $projectPath 'src-v3\processed\report\RADAR-LN3-V26h-l1-train-only-mahalanobis-ood-diagnostic.json'
$detailPath = Join-Path $projectPath 'src-v3\processed\report\RADAR-LN3-V26h-l1-train-only-mahalanobis-ood-diagnostic-parent-details.csv'
$maxReportBytes = 8 * 1024 * 1024
$maxDetailBytes = 2 * 1024 * 1024

function Invoke-RadarPython {
    param([Parameter(Mandatory = $true)][string[]]$Arguments)
    $runtimeArguments = @('-X', 'utf8') + $Arguments
    $display = $python + ' ' + ($runtimeArguments -join ' ')
    Write-Host ('> ' + $display) -ForegroundColor Cyan
    $code = 1
    Push-Location -LiteralPath $srcRoot
    try {
        & $python @runtimeArguments | Out-Host
        $code = [int]$LASTEXITCODE
    }
    finally {
        Pop-Location
    }
    if ($code -ne 0) {
        throw ('Python command failed with exit code ' + $code + ': ' + $display)
    }
}

function Assert-File {
    param([Parameter(Mandatory = $true)][string]$Path, [Parameter(Mandatory = $true)][string]$Description)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw ($Description + ' is missing: ' + $Path)
    }
}

function Read-Utf8Json {
    param([Parameter(Mandatory = $true)][string]$Path)
    Assert-File -Path $Path -Description 'JSON artifact'
    return ConvertFrom-Json -InputObject ([System.IO.File]::ReadAllText($Path, [System.Text.UTF8Encoding]::new($false, $true)))
}

function Get-Sha256 {
    param([Parameter(Mandatory = $true)][string]$Path)
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Assert-Prepared {
    Assert-File -Path $contractPath -Description 'V26h contract'
    Assert-File -Path $readinessPath -Description 'V26h readiness'
    $contract = Read-Utf8Json -Path $contractPath
    $readiness = Read-Utf8Json -Path $readinessPath
    if ([string]$contract.schema_version -ne 'RADAR-LN3-V26h-l1-train-only-mahalanobis-ood-diagnostic-contract-v1' -or
        [string]$readiness.schema_version -ne 'RADAR-LN3-V26h-l1-train-only-mahalanobis-ood-diagnostic-readiness-v1' -or
        [string]$readiness.status -ne 'passed' -or
        [string]$readiness.contract_sha256 -ne (Get-Sha256 -Path $contractPath) -or
        [int]$contract.fold_count -ne 5 -or
        [int]$contract.unknown_reference_fit_exposure -ne 0 -or
        [int]$contract.unknown_threshold_fit_exposure -ne 0 -or
        [bool]$contract.new_training -or
        [bool]$contract.new_threshold_fit -or
        [bool]$contract.semantic_action_mutation -or
        [bool]$contract.deployment_eligible) {
        throw 'V26h preparation artifacts violate the frozen train-only OOD diagnostic contract.'
    }
    $checks = @($readiness.checks.PSObject.Properties)
    if ($checks.Count -eq 0 -or @($checks | Where-Object { $_.Value -ne $true }).Count -ne 0) {
        throw 'V26h readiness has a failed or missing check.'
    }
}

function Invoke-Prepare {
    Invoke-RadarPython @(
        '-m', 'py_compile',
        'radar_ln3\known_gate_v26h.py',
        'build\known_gate_v26h.py',
        'report\known_gate_v26h.py',
        'smoke\known_gate_v26h_contract_test.py'
    )
    Invoke-RadarPython @('-m', 'build.known_gate_v26h', '--project-root', $projectPath)
    Assert-Prepared
}

function Invoke-Smoke {
    Assert-Prepared
    Invoke-RadarPython @('-m', 'smoke.known_gate_v26h_contract_test', '--project-root', $projectPath)
}

function Invoke-Diagnose {
    Assert-Prepared
    Invoke-RadarPython @('-m', 'report.known_gate_v26h', '--project-root', $projectPath)
    Assert-File -Path $reportPath -Description 'V26h ranking report'
    Assert-File -Path $detailPath -Description 'V26h parent detail CSV'
    if ((Get-Item -LiteralPath $reportPath).Length -gt $maxReportBytes) { throw ('V26h report exceeds bounded size: ' + $reportPath) }
    if ((Get-Item -LiteralPath $detailPath).Length -gt $maxDetailBytes) { throw ('V26h parent detail exceeds bounded size: ' + $detailPath) }
    $report = Read-Utf8Json -Path $reportPath
    if ([string]$report.status -ne 'completed_v26h_train_only_mahalanobis_ood_ranking_diagnostic' -or
        [string]$report.schema_version -ne 'RADAR-LN3-V26h-l1-train-only-mahalanobis-ood-diagnostic-v1' -or
        [int]$report.source_protocol.unknown_reference_fit_exposure -ne 0 -or
        [int]$report.source_protocol.unknown_threshold_fit_exposure -ne 0 -or
        [bool]$report.deployment_eligible) {
        throw 'V26h report violates the frozen train-only OOD diagnostic scope.'
    }
}

try {
    Assert-File -Path $python -Description 'Project Python interpreter'
    Write-Host ('[V26h] project=' + $projectPath) -ForegroundColor Green
    Write-Host ('[V26h] stage=' + $Stage + '; frozen V26e L1 epoch-32 checkpoints with per-fold train-Known-only Mahalanobis references') -ForegroundColor Green
    Write-Host '[V26h] No training, cache rebuild, threshold fitting, L3 mutation, semantic action replay or deployment registry is executed.' -ForegroundColor Green
    switch ($Stage) {
        'prepare' { Invoke-Prepare }
        'smoke' { Invoke-Smoke }
        'diagnose' { Invoke-Diagnose }
        'all' { Invoke-Prepare; Invoke-Smoke; Invoke-Diagnose }
    }
    Write-Host '[V26h] requested stage completed.' -ForegroundColor Green
}
catch {
    [Console]::Error.WriteLine('[V26h] pipeline stopped: ' + $_.Exception.Message)
    exit 1
}
