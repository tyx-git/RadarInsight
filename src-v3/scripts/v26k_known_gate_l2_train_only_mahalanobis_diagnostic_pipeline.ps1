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
$contractPath = Join-Path $projectPath 'data\processed\RADAR-LN3-V26k-cq08-l2-train-only-mahalanobis-knownness-diagnostic-contract.json'
$readinessPath = Join-Path $projectPath 'src-v3\processed\report\RADAR-LN3-V26k-l2-train-only-mahalanobis-knownness-diagnostic-readiness.json'
$reportPath = Join-Path $projectPath 'src-v3\processed\report\RADAR-LN3-V26k-l2-train-only-mahalanobis-knownness-diagnostic.json'
$detailPath = Join-Path $projectPath 'src-v3\processed\report\RADAR-LN3-V26k-l2-train-only-mahalanobis-knownness-diagnostic-parent-details.csv'
$maxReportBytes = 8 * 1024 * 1024
$maxDetailBytes = 4 * 1024 * 1024

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
    $stream = [System.IO.File]::OpenRead($Path)
    $sha256 = [System.Security.Cryptography.SHA256]::Create()
    try {
        return ([System.BitConverter]::ToString($sha256.ComputeHash($stream))).Replace('-', '').ToLowerInvariant()
    }
    finally {
        $sha256.Dispose()
        $stream.Dispose()
    }
}

function Test-PreparedCurrent {
    if (-not (Test-Path -LiteralPath $contractPath -PathType Leaf) -or
        -not (Test-Path -LiteralPath $readinessPath -PathType Leaf)) {
        return $false
    }
    $validationProgram = 'from pathlib import Path; from radar_ln3.known_gate_v26k import validate_prepared_contract; validate_prepared_contract(Path.cwd().parent)'
    Push-Location -LiteralPath $srcRoot
    try {
        try {
            & $python -X utf8 -c $validationProgram *> $null
            return ([int]$LASTEXITCODE -eq 0)
        }
        catch {
            return $false
        }
    }
    finally {
        Pop-Location
    }
}

function Archive-StalePreparation {
    $stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
    foreach ($artifactPath in @($contractPath, $readinessPath)) {
        if (Test-Path -LiteralPath $artifactPath -PathType Leaf) {
            $archivePath = $artifactPath + '.stale-before-v26k-regeneration-' + $stamp
            Move-Item -LiteralPath $artifactPath -Destination $archivePath -ErrorAction Stop
            Write-Host ('[V26k] archived stale preparation artifact to ' + $archivePath) -ForegroundColor Yellow
        }
    }
}

function Assert-Prepared {
    $contract = Read-Utf8Json -Path $contractPath
    $readiness = Read-Utf8Json -Path $readinessPath
    if ([string]$contract.schema_version -ne 'RADAR-LN3-V26k-l2-train-only-mahalanobis-knownness-diagnostic-contract-v1' -or
        [string]$readiness.schema_version -ne 'RADAR-LN3-V26k-l2-train-only-mahalanobis-knownness-diagnostic-readiness-v1' -or
        [string]$readiness.status -ne 'passed' -or
        [string]$readiness.contract_sha256 -ne (Get-Sha256 -Path $contractPath) -or
        [int]$contract.fold_count -ne 5 -or
        [int]$contract.fixed_l2_epoch -ne 15 -or
        [int]$contract.unknown_training_exposure -ne 0 -or
        [int]$contract.unknown_normalization_exposure -ne 0 -or
        [int]$contract.unknown_reference_fit_exposure -ne 0 -or
        [int]$contract.unknown_threshold_fit_exposure -ne 0 -or
        [int]$contract.unknown_checkpoint_selection_exposure -ne 0 -or
        [bool]$contract.new_training -or
        [bool]$contract.new_threshold_fit -or
        [bool]$contract.product_threshold_mutation -or
        [bool]$contract.semantic_action_mutation -or
        [bool]$contract.deployment_eligible -or
        [bool]$readiness.deployment_eligible) {
        throw 'V26k preparation artifacts violate the frozen train-only diagnostic contract.'
    }
    $checks = @($readiness.checks.PSObject.Properties)
    if ($checks.Count -eq 0 -or @($checks | Where-Object { $_.Value -ne $true }).Count -ne 0) {
        throw 'V26k readiness has a failed or missing check.'
    }
}

function Invoke-Prepare {
    Invoke-RadarPython @('-m', 'py_compile',
        'radar_ln3\known_gate_v26k.py',
        'build\known_gate_v26k.py',
        'report\known_gate_v26k.py',
        'smoke\known_gate_v26k_contract_test.py')
    if (-not (Test-PreparedCurrent) -and
        ((Test-Path -LiteralPath $contractPath -PathType Leaf) -or
         (Test-Path -LiteralPath $readinessPath -PathType Leaf))) {
        Archive-StalePreparation
    }
    Invoke-RadarPython @('-m', 'build.known_gate_v26k', '--project-root', $projectPath)
    Assert-Prepared
}

function Invoke-Smoke {
    Assert-Prepared
    Invoke-RadarPython @('-m', 'smoke.known_gate_v26k_contract_test', '--project-root', $projectPath)
}

function Invoke-Diagnose {
    Assert-Prepared
    Invoke-RadarPython @('-m', 'report.known_gate_v26k', '--project-root', $projectPath)
    Assert-File -Path $reportPath -Description 'V26k diagnostic report'
    Assert-File -Path $detailPath -Description 'V26k parent detail CSV'
    if ((Get-Item -LiteralPath $reportPath).Length -gt $maxReportBytes) {
        throw ('V26k report exceeds bounded size: ' + $reportPath)
    }
    if ((Get-Item -LiteralPath $detailPath).Length -gt $maxDetailBytes) {
        throw ('V26k parent detail CSV exceeds bounded size: ' + $detailPath)
    }
    $report = Read-Utf8Json -Path $reportPath
    if ([string]$report.status -ne 'completed_v26k_l2_train_only_mahalanobis_knownness_diagnostic' -or
        [string]$report.schema_version -ne 'RADAR-LN3-V26k-l2-train-only-mahalanobis-knownness-diagnostic-v1' -or
        [int]$report.unknown_protocol.unknown_reference_fit_exposure -ne 0 -or
        [int]$report.unknown_protocol.unknown_threshold_fit_exposure -ne 0 -or
        [bool]$report.deployment_eligible) {
        throw 'V26k report violates the frozen train-only diagnostic scope.'
    }
}

try {
    Assert-File -Path $python -Description 'Project Python interpreter'
    Write-Host ('[V26k] project=' + $projectPath) -ForegroundColor Green
    Write-Host ('[V26k] stage=' + $Stage + '; frozen V26c L2 epoch-15 train-only DBB embeddings with frozen V26f mixed audit') -ForegroundColor Green
    Write-Host '[V26k] No training, cache rebuild, L1 forward, L3 mutation, product calibration, semantic action replay or deployment is executed.' -ForegroundColor Green
    switch ($Stage) {
        'prepare' { Invoke-Prepare }
        'smoke' { Invoke-Smoke }
        'diagnose' { Invoke-Diagnose }
        'all' { Invoke-Prepare; Invoke-Smoke; Invoke-Diagnose }
    }
    Write-Host '[V26k] requested stage completed.' -ForegroundColor Green
}
catch {
    [Console]::Error.WriteLine('[V26k] pipeline stopped: ' + $_.Exception.Message)
    exit 1
}
