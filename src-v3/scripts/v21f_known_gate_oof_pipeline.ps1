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
    'RADAR-LN3-V21f-known-gate-course-interframe-oof-training-readiness.json'
)
$pairedReportPath = Join-Path $projectPath (
    'src-v3\processed\report\' +
    'RADAR-LN3-V21f-known-gate-course-interframe-paired-oof.json'
)
$readinessSchema = 'RADAR-LN3-V21f-known-gate-course-interframe-oof-readiness-v1'
$predictionSchema = 'RADAR-LN3-V21f-known-gate-course-interframe-oof-prediction-v1'
$pairedReportSchema = 'RADAR-LN3-V21f-known-gate-course-interframe-paired-oof-report-v1'
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
    param([Parameter(Mandatory = $true)][int]$Fold)
    return Join-Path $projectPath (
        'artifacts\RADAR-LN3-V21f-cq08-known-gate-course-interframe-paired-oof\' +
        'fold-' + $Fold.ToString('00') + '\treatment'
    )
}

function Assert-Readiness {
    Assert-File $readinessPath 'V21f OOF readiness'
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
        throw ('V21f OOF readiness is stale or failed: ' + $readinessPath)
    }
    return $readiness
}

function Assert-CompletedFold {
    param([Parameter(Mandatory = $true)][int]$Fold)
    $artifact = Get-ArtifactPath $Fold
    $predictionPath = Join-Path $artifact 'held-out-predictions.json'
    $summaryPath = Join-Path $artifact 'training-summary.json'
    $checkpointPath = Join-Path $artifact 'final_model.pt'
    Assert-File $predictionPath ('V21f fold ' + $Fold + ' prediction')
    Assert-File $summaryPath ('V21f fold ' + $Fold + ' summary')
    Assert-File $checkpointPath ('V21f fold ' + $Fold + ' final checkpoint')
    $prediction = Read-Utf8Json $predictionPath
    $summary = Read-Utf8Json $summaryPath
    $pairing = $prediction.paired_initialization
    if (
        $prediction.status -ne 'passed' -or
        $prediction.schema_version -ne $predictionSchema -or
        $prediction.experiment_name -ne 'V21f OOF' -or
        $prediction.arm -ne 'treatment' -or
        [int]$prediction.fold_id -ne $Fold -or
        [int]$prediction.epochs -ne 20 -or
        $prediction.held_out_checkpoint_selection -ne $false -or
        [int]$prediction.held_out_forward_count_before_final_epoch -ne 0 -or
        [int]$prediction.held_out_prediction_pass_count -ne 1 -or
        [int]$prediction.unknown_training_exposure -ne 0 -or
        [int]$prediction.validation_access_count -ne 0 -or
        [int]$prediction.test_payload_access_count -ne 0 -or
        @($prediction.records.parent).Count -ne [int]$prediction.held_out_scope.parent_count -or
        [int]$pairing.baseline_evidence_dim -ne 1 -or
        [int]$pairing.treatment_evidence_dim -ne 7 -or
        [int]$pairing.new_fusion_columns -ne 6 -or
        $pairing.new_fusion_columns_exactly_zero -ne $true -or
        $pairing.initial_logits_exactly_equal -ne $true -or
        [string]::IsNullOrWhiteSpace([string]$prediction.paired_baseline_reference_state_sha256) -or
        $summary.status -ne 'completed' -or
        [int]$summary.epoch -ne 20 -or
        $summary.deployment_eligible -ne $false
    ) {
        throw ('V21f fold ' + $Fold + ' is incomplete or outside the frozen OOF contract')
    }
}

function Assert-AllFoldsRequested {
    $normalized = @($Folds | Sort-Object -Unique)
    if (($normalized -join ',') -ne ($allFolds -join ',')) {
        throw 'This V21f stage requires all folds 0,1,2,3,4.'
    }
}

function Invoke-Prepare {
    Write-Host '[V21f OOF] preparing nested V21e-plus-spectral five-fold contract' -ForegroundColor Green
    Invoke-RadarPython @(
        '-m', 'py_compile',
        'radar_ln3\known_gate_features.py',
        'radar_ln3\known_gate.py',
        'radar_ln3\known_gate_v21e_oof.py',
        'radar_ln3\known_gate_v21f_oof.py',
        'build\known_gate_v21e_oof.py',
        'build\known_gate_v21f_oof.py',
        'train\known_gate_v21e_oof.py',
        'train\known_gate_v21f_oof.py',
        'report\known_gate_v21e_oof.py',
        'report\known_gate_v21f_oof.py',
        'smoke\known_gate_v21e_oof_contract_test.py',
        'smoke\known_gate_v21f_oof_contract_test.py'
    )
    Invoke-RadarPython @(
        '-m', 'build.known_gate_v21f_oof',
        '--project-root', $projectPath
    )
    $null = Assert-Readiness
    Invoke-RadarPython @(
        '-m', 'smoke.known_gate_v21f_oof_contract_test',
        '--project-root', $projectPath
    )
    Invoke-RadarPython @(
        '-m', 'smoke.static_isolation_check',
        '--root', $srcRoot
    )
}

function Invoke-FoldTraining {
    param([Parameter(Mandatory = $true)][int]$Fold)
    $null = Assert-Readiness
    $artifact = Get-ArtifactPath $Fold
    $predictionPath = Join-Path $artifact 'held-out-predictions.json'
    $lastCheckpoint = Join-Path $artifact 'last_model.pt'
    if (Test-Path -LiteralPath $predictionPath -PathType Leaf) {
        Assert-CompletedFold $Fold
        Write-Host ('[V21f OOF] treatment fold ' + $Fold + ' already completed.') -ForegroundColor Yellow
        return
    }
    $arguments = @(
        '-m', 'train.known_gate_v21f_oof',
        '--project-root', $projectPath,
        '--fold-id', [string]$Fold
    )
    if (Test-NonEmptyDirectory $artifact) {
        Assert-File $lastCheckpoint ('V21f fold ' + $Fold + ' resume checkpoint')
        $arguments += @('--resume-checkpoint', $lastCheckpoint)
        Write-Host ('[V21f OOF] resuming treatment fold ' + $Fold) -ForegroundColor Yellow
    }
    Invoke-RadarPython $arguments
    Assert-CompletedFold $Fold
}

function Invoke-Treatment {
    foreach ($fold in @($Folds | Sort-Object -Unique)) {
        Write-Host ('[V21f OOF] full 20-epoch Treatment fold ' + $fold) -ForegroundColor Green
        Invoke-FoldTraining $fold
    }
    if (@($Folds | Sort-Object -Unique).Count -ne 5) {
        Write-Host '[V21f OOF] partial Treatment folds completed; paired report waits for all five.' -ForegroundColor Yellow
    }
}

function Invoke-Evaluate {
    Assert-AllFoldsRequested
    foreach ($fold in $allFolds) {
        Assert-CompletedFold $fold
    }
    Invoke-RadarPython @(
        '-m', 'report.known_gate_v21f_oof',
        '--project-root', $projectPath
    )
    Assert-File $pairedReportPath 'V21f paired OOF report'
    $report = Read-Utf8Json $pairedReportPath
    if (
        $report.schema_version -ne $pairedReportSchema -or
        $report.deployment_eligible -ne $false -or
        $report.decision.online_known_gate_adoption_allowed -ne $false -or
        $report.decision.advance_to_threshold_calibration -ne $false
    ) {
        throw ('V21f paired report violates its exploratory contract: ' + $pairedReportPath)
    }
}

try {
    Write-Host ('[V21f OOF] project=' + $projectPath) -ForegroundColor Green
    Write-Host ('[V21f OOF] stage=' + $Stage + '; folds=' + (($Folds | Sort-Object -Unique) -join ',')) -ForegroundColor Green
    switch ($Stage) {
        'prepare' { Invoke-Prepare }
        'treatment' { Invoke-Treatment }
        'evaluate' { Invoke-Evaluate }
        'all' {
            Assert-AllFoldsRequested
            Invoke-Prepare
            Invoke-Treatment
            Invoke-Evaluate
        }
    }
    Write-Host '[V21f OOF] requested stages completed. Unknown audit, threshold fitting, test, rejection, stable and P95 remain unexecuted.' -ForegroundColor Green
}
catch {
    [Console]::Error.WriteLine('[V21f OOF] pipeline stopped: ' + $_.Exception.Message)
    exit 1
}
