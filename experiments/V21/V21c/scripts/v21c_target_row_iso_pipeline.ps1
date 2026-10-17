[CmdletBinding()]
param(
    [ValidateSet('prepare', 'train', 'evaluate', 'all')]
    [string]$Stage = 'prepare'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$srcRoot = Split-Path -Parent $scriptRoot
$projectRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $srcRoot))
$python = Join-Path $projectRoot '.venv\Scripts\python.exe'
$config = Join-Path $projectRoot 'config\RADAR-LN3\RADAR-LN3-V21c-cq08-liquid-known-gate-target-row-iso-clutter-drone-seed43-config.json'
$artifact = Join-Path $projectRoot 'artifacts\RADAR-LN3-V21c-cq08-liquid-known-gate-target-row-iso-clutter-drone-seed43-formal'
$readiness = Join-Path $projectRoot 'src-v3\processed\report\RADAR-LN3-V21c-known-gate-target-row-iso-clutter-drone-training-readiness.json'
$report = Join-Path $projectRoot 'src-v3\processed\report\RADAR-LN3-V21c-known-gate-target-row-iso-clutter-drone-paired-validation.json'
$notExecuted = -join ([char[]](0x672A, 0x6267, 0x884C))

function Invoke-RadarPython([string[]]$Arguments) {
    Write-Host ('> ' + $python + ' ' + ($Arguments -join ' ')) -ForegroundColor Cyan
    Push-Location $srcRoot
    try {
        & $python @Arguments
    }
    finally {
        Pop-Location
    }
    if ($LASTEXITCODE -ne 0) {
        throw ('Python command failed with exit code ' + $LASTEXITCODE)
    }
}

function Assert-File([string]$Path, [string]$Description) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw ($Description + ' is missing: ' + $Path)
    }
}

function Assert-Readiness {
    Assert-File $readiness 'V21c readiness'
    Assert-File $config 'V21c config'
    $payload = Get-Content -LiteralPath $readiness -Raw -Encoding UTF8 | ConvertFrom-Json
    $actualConfigHash = (Get-FileHash -LiteralPath $config -Algorithm SHA256).Hash.ToLowerInvariant()
    $failed = @(
        $payload.checks.PSObject.Properties |
            Where-Object { $_.Value -ne $true } |
            ForEach-Object { $_.Name }
    )
    if (
        $payload.status -ne 'passed' -or
        $payload.schema_version -ne 'RADAR-LN3-V21c-known-gate-target-row-iso-clutter-drone-training-readiness-v1' -or
        $payload.config.sha256 -ne $actualConfigHash -or
        $payload.selected_baseline.seed -ne 43 -or
        $payload.validation_used_for_readiness_gate -ne $false -or
        $payload.validation_feature_values_used_for_readiness_gate -ne $false -or
        $payload.test_payload_access_count -ne 0 -or
        $payload.formal_training_at_report_generation -ne $notExecuted -or
        $payload.test -ne $notExecuted -or
        $payload.deployment_eligible -ne $false -or
        $failed.Count -ne 0
    ) {
        throw ('V21c readiness is stale, failed or outside the sealed contract: ' + $readiness)
    }
}

function Assert-Completed {
    $summaryPath = Join-Path $artifact 'training-summary.json'
    Assert-File $summaryPath 'V21c training summary'
    $summary = Get-Content -LiteralPath $summaryPath -Raw -Encoding UTF8 | ConvertFrom-Json
    if (
        $summary.status -ne 'completed' -or
        $summary.task_name -ne 'known_gate' -or
        $summary.variant -ne 'v21c' -or
        $summary.seed -ne 43 -or
        $summary.train_parent_count -ne 986 -or
        $summary.validation_parent_count -ne 232 -or
        $summary.validation_clutter_support -ne 9 -or
        $summary.history.Count -ne 20 -or
        $summary.full_parameter_training -ne $true -or
        $summary.all_model_parameters_trainable -ne $true -or
        $summary.model_state_changed_from_initialization -ne $true -or
        $summary.paired_initialization.paired_baseline -ne $true -or
        $summary.paired_initialization.shared_parameters_and_buffers_exactly_equal -ne $true -or
        $summary.paired_initialization.new_fusion_columns -ne 1 -or
        $summary.paired_initialization.new_fusion_columns_exactly_zero -ne $true -or
        $summary.paired_initialization.initial_logits_exactly_equal -ne $true -or
        $summary.test_evaluation -ne 'not_executed' -or
        $summary.oof_threshold_calibration -ne 'not_executed' -or
        $summary.full_cascade_evaluation -ne 'not_executed' -or
        $summary.formal_rejection -ne 'not_executed' -or
        $summary.stable_evaluation -ne 'not_executed' -or
        $summary.p95_latency -ne 'not_executed' -or
        $summary.deployment_eligible -ne $false
    ) {
        throw ('V21c is not a completed full selected-seed paired run: ' + $summaryPath)
    }
    Assert-File (Join-Path $artifact 'best_model.pt') 'V21c best checkpoint'
    Assert-File (Join-Path $artifact 'last_model.pt') 'V21c last checkpoint'
    Assert-File (Join-Path $artifact 'sample_detail_information.json') 'V21c sample detail'
    for ($epoch = 1; $epoch -le 20; $epoch++) {
        $name = 'epoch-' + $epoch.ToString('000') + '.pt'
        Assert-File (Join-Path $artifact ('epochs\' + $name)) ('V21c epoch ' + $epoch)
    }
}


function Invoke-Prepare {
    Invoke-RadarPython @(
        '-m', 'build.known_gate_v21c',
        '--project-root', $projectRoot
    )
    Assert-Readiness
}

function Move-PreCheckpointArtifact {
    $checkpointFiles = @(
        Get-ChildItem -LiteralPath $artifact -Recurse -File -Filter '*.pt' -ErrorAction SilentlyContinue
    )
    if ($checkpointFiles.Count -gt 0) {
        throw ('V21c artifact contains checkpoints but no resumable last_model.pt: ' + $artifact)
    }
    $archive = $artifact + '.interrupted-before-checkpoint-' + (Get-Date -Format 'yyyyMMdd-HHmmss')
    Move-Item -LiteralPath $artifact -Destination $archive
    Write-Host ('[V21c] archived pre-checkpoint artifact: ' + $archive) -ForegroundColor Yellow
}

function Invoke-Train {
    Assert-Readiness
    $summaryPath = Join-Path $artifact 'training-summary.json'
    if (Test-Path -LiteralPath $summaryPath -PathType Leaf) {
        $summary = Get-Content -LiteralPath $summaryPath -Raw -Encoding UTF8 | ConvertFrom-Json
        if ($summary.status -eq 'completed') {
            Assert-Completed
            Write-Host '[V21c] formal training already completed; preserving artifact.' -ForegroundColor Yellow
            return
        }
        if ($summary.status -ne 'failed') {
            throw ('V21c summary has an unsupported status: ' + $summaryPath)
        }
    }
    $arguments = @(
        '-m', 'train.known_gate_v21',
        '--project-root', $projectRoot,
        '--config', $config
    )
    $lastCheckpoint = Join-Path $artifact 'last_model.pt'
    if (Test-Path -LiteralPath $artifact -PathType Container) {
        $items = @(Get-ChildItem -LiteralPath $artifact -Force)
        if ($items.Count -gt 0) {
            if (Test-Path -LiteralPath $lastCheckpoint -PathType Leaf) {
                $arguments += @('--resume-checkpoint', $lastCheckpoint)
                Write-Host ('[V21c] resuming from ' + $lastCheckpoint) -ForegroundColor Yellow
            }
            else {
                Move-PreCheckpointArtifact
            }
        }
    }
    Invoke-RadarPython $arguments
    Assert-Completed
}

function Invoke-Evaluate {
    Assert-Readiness
    Assert-Completed
    Invoke-RadarPython @(
        '-m', 'report.known_gate_v21c',
        '--project-root', $projectRoot
    )
    Assert-File $report 'V21c paired validation report'
    $payload = Get-Content -LiteralPath $report -Raw -Encoding UTF8 | ConvertFrom-Json
    if (
        $payload.status -ne 'completed' -or
        $payload.paired_protocol.seed -ne 43 -or
        $payload.paired_protocol.all_20_epoch_parent_order_hashes_equal -ne $true -or
        $payload.primary_hypothesis.labels.Count -ne 2 -or
        $payload.primary_hypothesis.labels[0] -ne 'DroneTarget' -or
        $payload.primary_hypothesis.labels[1] -ne 'ClutterTarget' -or
        $payload.safety_labels.Count -ne 2 -or
        $payload.safety_labels[0] -ne 'BirdTarget' -or
        $payload.safety_labels[1] -ne 'BalloonTarget' -or
        $payload.bootstrap.replicates -ne 5000 -or
        $payload.trained_iso_fusion_column.element_count -ne 48 -or
        $payload.trained_iso_fusion_column.nonzero_element_count -ne 48 -or
        $payload.clutter_support_warning.validation_parent_support -ne 9 -or
        $payload.test -ne $notExecuted -or
        $payload.OOF -ne $notExecuted -or
        $payload.full_cascade -ne $notExecuted -or
        $payload.formal_rejection -ne $notExecuted -or
        $payload.stable -ne $notExecuted -or
        $payload.P95 -ne $notExecuted -or
        $payload.deployment_eligible -ne $false
    ) {
        throw ('V21c report violates the frozen local/test-sealed contract: ' + $report)
    }
}

try {
    Assert-File $python 'Project Python interpreter'
    Write-Host ('[V21c] project=' + $projectRoot) -ForegroundColor Green
    Write-Host ('[V21c] stage=' + $Stage + '; seed=43; feature=target-row iso; primary=Clutter/Drone only') -ForegroundColor Green
    switch ($Stage) {
        'prepare' { Invoke-Prepare }
        'train' { Invoke-Train }
        'evaluate' { Invoke-Evaluate }
        'all' {
            Invoke-Prepare
            Invoke-Train
            Invoke-Evaluate
        }
    }
    Write-Host '[V21c] requested stage completed. Test, OOF, rejection, cascade, stable, P95 and deployment remain unexecuted.' -ForegroundColor Green
}
catch {
    Write-Error ('V21c target-row iso pipeline stopped: ' + $_.Exception.Message)
    exit 1
}
