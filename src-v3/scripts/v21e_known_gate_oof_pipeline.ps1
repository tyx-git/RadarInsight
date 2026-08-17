[CmdletBinding()]
param(
    [ValidateSet('prepare', 'treatment', 'control', 'evaluate', 'all')]
    [string]$Stage = 'all',
    [ValidateRange(0, 4)]
    [int[]]$Folds = @(0, 1, 2, 3, 4),
    [string]$ProjectRoot = ''
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$utf8NoBom = [System.Text.UTF8Encoding]::new($false)
[Console]::InputEncoding = $utf8NoBom
[Console]::OutputEncoding = $utf8NoBom
$OutputEncoding = $utf8NoBom

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$srcRoot = Split-Path -Parent $scriptRoot
$defaultProjectRoot = Split-Path -Parent $srcRoot
$projectPath = [System.IO.Path]::GetFullPath(
    $(if ([string]::IsNullOrWhiteSpace($ProjectRoot)) { $defaultProjectRoot } else { $ProjectRoot })
)
$python = Join-Path $projectPath '.venv\Scripts\python.exe'
$readinessPath = Join-Path $projectPath (
    'src-v3\processed\report\' +
    'RADAR-LN3-V21e-known-gate-capture-content-grouped-oof-training-readiness.json'
)
$treatmentReportPath = Join-Path $projectPath (
    'src-v3\processed\report\' +
    'RADAR-LN3-V21e-known-gate-capture-content-grouped-oof-treatment.json'
)
$pairedReportPath = Join-Path $projectPath (
    'src-v3\processed\report\' +
    'RADAR-LN3-V21e-known-gate-capture-content-grouped-oof-paired.json'
)
$readinessSchema = 'RADAR-LN3-V21e-known-gate-capture-content-grouped-oof-readiness-v1'
$predictionSchema = 'RADAR-LN3-V21e-known-gate-capture-content-grouped-oof-prediction-v1'
$treatmentReportSchema = 'RADAR-LN3-V21e-known-gate-capture-content-grouped-oof-treatment-report-v1'
$pairedReportSchema = 'RADAR-LN3-V21e-known-gate-capture-content-grouped-oof-paired-report-v1'
$allFolds = @(0, 1, 2, 3, 4)

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw ('Project Python interpreter is missing: ' + $python)
}

function Invoke-RadarPython {
    param([Parameter(Mandatory = $true)][string[]]$Arguments)

    $display = $python + ' ' + ($Arguments -join ' ')
    Write-Host ('> ' + $display) -ForegroundColor Cyan
    $exitCode = 0
    Push-Location $srcRoot
    try {
        & $python @Arguments | Out-Host
        $exitCode = [int]$LASTEXITCODE
    }
    finally {
        Pop-Location
    }
    if ($exitCode -ne 0) {
        throw ('Python command failed with exit code ' + $exitCode + ': ' + $display)
    }
}

function Read-Utf8Json {
    param([Parameter(Mandatory = $true)][string]$Path)
    $text = [System.IO.File]::ReadAllText(
        $Path,
        [System.Text.UTF8Encoding]::new($false, $true)
    )
    return ConvertFrom-Json -InputObject $text
}

function Assert-File {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Description
    )
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw ($Description + ' is missing: ' + $Path)
    }
}

function Test-NonEmptyDirectory {
    param([Parameter(Mandatory = $true)][string]$Path)
    if (-not (Test-Path -LiteralPath $Path -PathType Container)) {
        return $false
    }
    return $null -ne (Get-ChildItem -LiteralPath $Path -Force | Select-Object -First 1)
}

function Get-ArtifactPath {
    param(
        [Parameter(Mandatory = $true)][ValidateSet('control', 'treatment')][string]$Arm,
        [Parameter(Mandatory = $true)][int]$Fold
    )
    return Join-Path $projectPath (
        'artifacts\RADAR-LN3-V21e-cq08-known-gate-causal-course-paired-oof\' +
        'fold-' + $Fold.ToString('00') + '\' + $Arm
    )
}

function Assert-Readiness {
    Assert-File $readinessPath 'V21e OOF readiness'
    $readiness = Read-Utf8Json $readinessPath
    $failedChecks = @(
        $readiness.checks.PSObject.Properties |
            Where-Object { $_.Value -ne $true }
    )
    if (
        $readiness.status -ne 'passed' -or
        $readiness.schema_version -ne $readinessSchema -or
        [int]$readiness.fold_count -ne 5 -or
        [int]$readiness.known_parent_count -ne 986 -or
        [int]$readiness.provenance_component_count -ne 975 -or
        [int]$readiness.selected_seed -ne 43 -or
        $readiness.validation_used_for_readiness_gate -ne $false -or
        [int]$readiness.test_payload_access_count -ne 0 -or
        $readiness.deployment_eligible -ne $false -or
        $failedChecks.Count -ne 0
    ) {
        throw ('V21e OOF readiness is stale or failed: ' + $readinessPath)
    }
    return $readiness
}

function Assert-CompletedFold {
    param(
        [Parameter(Mandatory = $true)][ValidateSet('control', 'treatment')][string]$Arm,
        [Parameter(Mandatory = $true)][int]$Fold
    )
    $artifact = Get-ArtifactPath $Arm $Fold
    $predictionPath = Join-Path $artifact 'held-out-predictions.json'
    $summaryPath = Join-Path $artifact 'training-summary.json'
    $checkpointPath = Join-Path $artifact 'final_model.pt'
    Assert-File $predictionPath ($Arm + ' fold ' + $Fold + ' prediction')
    Assert-File $summaryPath ($Arm + ' fold ' + $Fold + ' summary')
    Assert-File $checkpointPath ($Arm + ' fold ' + $Fold + ' final checkpoint')
    $prediction = Read-Utf8Json $predictionPath
    $summary = Read-Utf8Json $summaryPath
    if (
        $prediction.status -ne 'passed' -or
        $prediction.schema_version -ne $predictionSchema -or
        $prediction.arm -ne $Arm -or
        [int]$prediction.fold_id -ne $Fold -or
        [int]$prediction.epochs -ne 20 -or
        $prediction.held_out_checkpoint_selection -ne $false -or
        [int]$prediction.held_out_forward_count_before_final_epoch -ne 0 -or
        [int]$prediction.held_out_prediction_pass_count -ne 1 -or
        [int]$prediction.unknown_training_exposure -ne 0 -or
        [int]$prediction.validation_access_count -ne 0 -or
        [int]$prediction.test_payload_access_count -ne 0 -or
        @($prediction.records.parent).Count -ne [int]$prediction.held_out_scope.parent_count -or
        $summary.status -ne 'completed' -or
        [int]$summary.epoch -ne 20 -or
        $summary.deployment_eligible -ne $false
    ) {
        throw ($Arm + ' fold ' + $Fold + ' is incomplete or outside the OOF contract')
    }
}

function Invoke-Prepare {
    Write-Host '[V21e OOF] preparing five provenance-grouped folds and fold-train normalization' -ForegroundColor Green
    Invoke-RadarPython @(
        '-m', 'py_compile',
        'radar_ln3\known_gate_features.py',
        'radar_ln3\known_gate.py',
        'radar_ln3\known_gate_v21e_oof.py',
        'build\known_gate_v21e_oof.py',
        'train\known_gate_v21e_oof.py',
        'report\known_gate_v21e_oof.py',
        'smoke\known_gate_v21e_oof_contract_test.py'
    )
    Invoke-RadarPython @(
        '-m', 'build.known_gate_v21e_oof',
        '--project-root', $projectPath
    )
    $null = Assert-Readiness
    Invoke-RadarPython @(
        '-m', 'smoke.known_gate_v21e_oof_contract_test',
        '--project-root', $projectPath
    )
    Invoke-RadarPython @(
        '-m', 'smoke.static_isolation_check',
        '--root', $srcRoot
    )
}

function Invoke-FoldTraining {
    param(
        [Parameter(Mandatory = $true)][ValidateSet('control', 'treatment')][string]$Arm,
        [Parameter(Mandatory = $true)][int]$Fold
    )
    $null = Assert-Readiness
    $artifact = Get-ArtifactPath $Arm $Fold
    $predictionPath = Join-Path $artifact 'held-out-predictions.json'
    $lastCheckpoint = Join-Path $artifact 'last_model.pt'
    if (Test-Path -LiteralPath $predictionPath -PathType Leaf) {
        Assert-CompletedFold $Arm $Fold
        Write-Host ('[V21e OOF] ' + $Arm + ' fold ' + $Fold + ' already completed.') -ForegroundColor Yellow
        return
    }
    $arguments = @(
        '-m', 'train.known_gate_v21e_oof',
        '--project-root', $projectPath,
        '--arm', $Arm,
        '--fold-id', [string]$Fold
    )
    if (Test-NonEmptyDirectory $artifact) {
        Assert-File $lastCheckpoint ($Arm + ' fold ' + $Fold + ' resume checkpoint')
        $arguments += @('--resume-checkpoint', $lastCheckpoint)
        Write-Host ('[V21e OOF] resuming ' + $Arm + ' fold ' + $Fold) -ForegroundColor Yellow
    }
    Invoke-RadarPython $arguments
    Assert-CompletedFold $Arm $Fold
}

function Assert-AllFoldsRequested {
    $normalized = @($Folds | Sort-Object -Unique)
    if (($normalized -join ',') -ne ($allFolds -join ',')) {
        throw 'This stage requires all folds 0,1,2,3,4.'
    }
}

function Invoke-Treatment {
    foreach ($fold in @($Folds | Sort-Object -Unique)) {
        Write-Host ('[V21e OOF] full 20-epoch Treatment fold ' + $fold) -ForegroundColor Green
        Invoke-FoldTraining 'treatment' $fold
    }
    if (@($Folds | Sort-Object -Unique).Count -ne 5) {
        Write-Host '[V21e OOF] partial Treatment folds completed; futility report waits for all five.' -ForegroundColor Yellow
        return $true
    }
    Invoke-RadarPython @(
        '-m', 'report.known_gate_v21e_oof',
        '--project-root', $projectPath,
        '--mode', 'treatment'
    )
    Assert-File $treatmentReportPath 'V21e OOF Treatment report'
    $report = Read-Utf8Json $treatmentReportPath
    if (
        $report.schema_version -ne $treatmentReportSchema -or
        $report.deployment_eligible -ne $false
    ) {
        throw ('V21e OOF Treatment report violates its contract: ' + $treatmentReportPath)
    }
    return [bool]$report.advance_to_control_training
}

function Invoke-Control {
    foreach ($fold in @($Folds | Sort-Object -Unique)) {
        Write-Host ('[V21e OOF] full 20-epoch Control fold ' + $fold) -ForegroundColor Green
        Invoke-FoldTraining 'control' $fold
    }
}

function Invoke-Evaluate {
    Assert-AllFoldsRequested
    foreach ($fold in $allFolds) {
        Assert-CompletedFold 'treatment' $fold
        Assert-CompletedFold 'control' $fold
    }
    Invoke-RadarPython @(
        '-m', 'report.known_gate_v21e_oof',
        '--project-root', $projectPath,
        '--mode', 'paired'
    )
    Assert-File $pairedReportPath 'V21e OOF paired report'
    $report = Read-Utf8Json $pairedReportPath
    if (
        $report.schema_version -ne $pairedReportSchema -or
        $report.deployment_eligible -ne $false -or
        $report.decision.online_known_gate_adoption_allowed -ne $false
    ) {
        throw ('V21e OOF paired report violates its contract: ' + $pairedReportPath)
    }
}

try {
    Write-Host ('[V21e OOF] project=' + $projectPath) -ForegroundColor Green
    Write-Host ('[V21e OOF] stage=' + $Stage + '; folds=' + (($Folds | Sort-Object -Unique) -join ',')) -ForegroundColor Green
    switch ($Stage) {
        'prepare' { Invoke-Prepare }
        'treatment' { $null = Invoke-Treatment }
        'control' { Invoke-Control }
        'evaluate' { Invoke-Evaluate }
        'all' {
            Assert-AllFoldsRequested
            Invoke-Prepare
            $advance = Invoke-Treatment
            if (-not $advance) {
                Write-Host '[V21e OOF] Treatment absolute guard failed; Control training is intentionally skipped.' -ForegroundColor Yellow
                break
            }
            Invoke-Control
            Invoke-Evaluate
        }
    }
    Write-Host '[V21e OOF] requested stages completed. Threshold fitting, Unknown audit, test, rejection, stable and P95 remain unexecuted.' -ForegroundColor Green
}
catch {
    [Console]::Error.WriteLine('[V21e OOF] pipeline stopped: ' + $_.Exception.Message)
    exit 1
}
