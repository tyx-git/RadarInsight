[CmdletBinding()]
param(
    [ValidateSet('prepare', 'smoke', 'baseline', 'v21a', 'v21b', 'v21c', 'v21d', 'train-all', 'evaluate', 'all')]
    [string]$Stage = 'all',

    [ValidateSet('all', 'v21a', 'v21b', 'v21c', 'v21d')]
    [string]$Variant = 'all'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$srcRoot = Split-Path -Parent $scriptRoot
$projectRoot = Split-Path -Parent $srcRoot
$python = Join-Path $projectRoot '.venv\Scripts\python.exe'
$configRoot = Join-Path $projectRoot 'config\RADAR-LN3'
$readinessPath = Join-Path $projectRoot 'src-v3\processed\report\RADAR-LN3-V21-known-gate-training-readiness.json'
$readinessSchema = 'RADAR-LN3-V21-known-gate-training-readiness-v1'
$notExecuted = -join ([char[]](0x672A, 0x6267, 0x884C))
$variants = @('baseline', 'v21a', 'v21b', 'v21c', 'v21d')

function Invoke-RadarPython([string[]]$Arguments) {
    $display = $python + ' ' + ($Arguments -join ' ')
    Write-Host ('> ' + $display) -ForegroundColor Cyan
    Push-Location $srcRoot
    try {
        & $python @Arguments
    }
    finally {
        Pop-Location
    }
    if ($LASTEXITCODE -ne 0) {
        throw ('Python command failed with exit code ' + $LASTEXITCODE + ': ' + $display)
    }
}

function Assert-File([string]$Path, [string]$Description) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw ($Description + ' is missing: ' + $Path)
    }
}

function Get-VariantVersion([string]$Name) {
    switch ($Name) {
        'baseline' { return 'V21' }
        'v21a' { return 'V21a' }
        'v21b' { return 'V21b' }
        'v21c' { return 'V21c' }
        'v21d' { return 'V21d' }
        default { throw ('Unsupported V21 variant: ' + $Name) }
    }
}

function Get-VariantSlug([string]$Name) {
    switch ($Name) {
        'baseline' { return 'baseline' }
        'v21a' { return 'spectral-temporal-consistency' }
        'v21b' { return 'modal-consistency' }
        'v21c' { return 'target-row-iso' }
        'v21d' { return 'd-range-stability' }
        default { throw ('Unsupported V21 variant: ' + $Name) }
    }
}

function Get-ConfigPath([string]$Name) {
    $version = Get-VariantVersion $Name
    $slug = Get-VariantSlug $Name
    return Join-Path $configRoot (
        'RADAR-LN3-' + $version + '-cq08-liquid-known-gate-' +
        $slug + '-seed42-config.json'
    )
}

function Get-ArtifactRoot([string]$Name) {
    $version = Get-VariantVersion $Name
    $slug = Get-VariantSlug $Name
    return Join-Path $projectRoot (
        'artifacts\RADAR-LN3-' + $version + '-cq08-liquid-known-gate-' +
        $slug + '-seed42-formal'
    )
}

function Assert-Readiness {
    Assert-File $readinessPath 'V21 KnownGate readiness report'
    $readiness = Get-Content -LiteralPath $readinessPath -Raw -Encoding UTF8 | ConvertFrom-Json
    if (
        $readiness.status -ne 'passed' -or
        $readiness.schema_version -ne $readinessSchema -or
        $readiness.validation_used_for_readiness_gate -ne $false -or
        $readiness.validation_feature_values_used_for_readiness_gate -ne $false -or
        $readiness.test_payload_access_count -ne 0 -or
        $readiness.formal_training_at_report_generation -ne $notExecuted -or
        $readiness.test -ne $notExecuted -or
        $readiness.deployment_eligible -ne $false
    ) {
        throw ('V21 readiness is not passed/train-only/test-sealed: ' + $readinessPath)
    }
    $failed = @(
        $readiness.checks.PSObject.Properties |
            Where-Object { $_.Value -ne $true } |
            ForEach-Object { $_.Name }
    )
    if ($failed.Count -gt 0) {
        throw ('V21 readiness contains failed checks: ' + ($failed -join ','))
    }
    foreach ($name in $variants) {
        $configPath = Get-ConfigPath $name
        Assert-File $configPath ('V21 ' + $name + ' config')
        $binding = $readiness.configs.$name
        $actualHash = (Get-FileHash -LiteralPath $configPath -Algorithm SHA256).Hash.ToLowerInvariant()
        if (
            $null -eq $binding -or
            [System.IO.Path]::GetFullPath([string]$binding.path) -ne [System.IO.Path]::GetFullPath($configPath) -or
            $binding.sha256 -ne $actualHash
        ) {
            throw ('V21 readiness does not bind config ' + $name + ': ' + $configPath)
        }
    }
}

function Assert-CompletedVariant([string]$Name) {
    $artifactRoot = Get-ArtifactRoot $Name
    $summaryPath = Join-Path $artifactRoot 'training-summary.json'
    Assert-File $summaryPath ('V21 ' + $Name + ' training summary')
    $summary = Get-Content -LiteralPath $summaryPath -Raw -Encoding UTF8 | ConvertFrom-Json
    foreach ($fieldName in @(
        'status', 'task_name', 'variant', 'best_epoch', 'train_parent_count',
        'validation_parent_count', 'validation_clutter_support',
        'full_parameter_training', 'all_model_parameters_trainable',
        'model_state_changed_from_initialization', 'test_evaluation',
        'oof_threshold_calibration', 'full_cascade_evaluation',
        'formal_rejection', 'stable_evaluation', 'p95_latency',
        'deployment_eligible'
    )) {
        if (-not ($summary.PSObject.Properties.Name -contains $fieldName)) {
            throw ('V21 ' + $Name + ' summary is missing field ' + $fieldName + ': ' + $summaryPath)
        }
    }
    if (
        $summary.status -ne 'completed' -or
        $summary.task_name -ne 'known_gate' -or
        $summary.variant -ne $Name -or
        $summary.train_parent_count -ne 986 -or
        $summary.validation_parent_count -ne 232 -or
        $summary.validation_clutter_support -ne 9 -or
        $summary.full_parameter_training -ne $true -or
        $summary.all_model_parameters_trainable -ne $true -or
        $summary.model_state_changed_from_initialization -ne $true
    ) {
        throw ('V21 ' + $Name + ' is not a completed full KnownGate run: ' + $summaryPath)
    }
    if (
        $summary.test_evaluation -ne 'not_executed' -or
        $summary.oof_threshold_calibration -ne 'not_executed' -or
        $summary.full_cascade_evaluation -ne 'not_executed' -or
        $summary.formal_rejection -ne 'not_executed' -or
        $summary.stable_evaluation -ne 'not_executed' -or
        $summary.p95_latency -ne 'not_executed' -or
        $summary.deployment_eligible -ne $false
    ) {
        throw ('V21 ' + $Name + ' summary violates the local/test-sealed contract: ' + $summaryPath)
    }
    if ($Name -ne 'baseline') {
        if (
            $summary.paired_initialization.paired_baseline -ne $true -or
            $summary.paired_initialization.shared_parameters_and_buffers_exactly_equal -ne $true -or
            $summary.paired_initialization.new_fusion_columns_exactly_zero -ne $true -or
            $summary.paired_initialization.initial_logits_exactly_equal -ne $true
        ) {
            throw ('V21 ' + $Name + ' does not prove paired Baseline initialization: ' + $summaryPath)
        }
    }
    Assert-File (Join-Path $artifactRoot 'best_model.pt') ('V21 ' + $Name + ' best checkpoint')
    Assert-File (Join-Path $artifactRoot 'last_model.pt') ('V21 ' + $Name + ' last checkpoint')
    Assert-File (Join-Path $artifactRoot 'sample_detail_information.json') ('V21 ' + $Name + ' sample detail')
    for ($epoch = 1; $epoch -le 20; $epoch++) {
        $epochName = 'epoch-' + $epoch.ToString('000') + '.pt'
        Assert-File (Join-Path $artifactRoot ('epochs\' + $epochName)) ('V21 ' + $Name + ' epoch checkpoint')
    }
}

function Invoke-Smoke {
    Invoke-RadarPython @(
        '-m', 'smoke.known_gate_v21_contract_test',
        '--project-root', $projectRoot
    )
    Invoke-RadarPython @(
        '-m', 'smoke.static_isolation_check',
        '--root', $srcRoot
    )
}

function Invoke-Prepare {
    Invoke-RadarPython @(
        '-m', 'build.known_gate_v21',
        '--project-root', $projectRoot
    )
    Invoke-Smoke
    Assert-Readiness
}

function Move-PreCheckpointArtifact([string]$Name, [string]$ArtifactRoot) {
    $checkpointFiles = @(
        Get-ChildItem -LiteralPath $ArtifactRoot -Recurse -File -Filter '*.pt' -ErrorAction SilentlyContinue
    )
    if ($checkpointFiles.Count -gt 0) {
        throw (
            'V21 ' + $Name + ' artifact contains checkpoint files but no resumable last_model.pt; inspect: ' +
            $ArtifactRoot
        )
    }
    $archive = $ArtifactRoot + '.interrupted-before-checkpoint-' + (Get-Date -Format 'yyyyMMdd-HHmmss')
    if (Test-Path -LiteralPath $archive) {
        throw ('V21 pre-checkpoint archive target already exists: ' + $archive)
    }
    Move-Item -LiteralPath $ArtifactRoot -Destination $archive
    Write-Host ('[V21] archived pre-checkpoint artifact: ' + $archive) -ForegroundColor Yellow
}

function Invoke-TrainVariant([string]$Name) {
    Assert-Readiness
    $configPath = Get-ConfigPath $Name
    $artifactRoot = Get-ArtifactRoot $Name
    $summaryPath = Join-Path $artifactRoot 'training-summary.json'
    if (Test-Path -LiteralPath $summaryPath -PathType Leaf) {
        $summary = Get-Content -LiteralPath $summaryPath -Raw -Encoding UTF8 | ConvertFrom-Json
        if ($summary.status -eq 'completed') {
            Assert-CompletedVariant $Name
            Write-Host ('[V21] ' + $Name + ' already completed; preserving formal artifact.') -ForegroundColor Yellow
            return
        }
        if ($summary.status -ne 'failed') {
            throw ('V21 ' + $Name + ' has an unsupported summary status: ' + $summaryPath)
        }
    }
    $arguments = @(
        '-m', 'train.known_gate_v21',
        '--project-root', $projectRoot,
        '--config', $configPath
    )
    $lastCheckpoint = Join-Path $artifactRoot 'last_model.pt'
    if (Test-Path -LiteralPath $artifactRoot -PathType Container) {
        $items = @(Get-ChildItem -LiteralPath $artifactRoot -Force)
        if ($items.Count -gt 0) {
            if (Test-Path -LiteralPath $lastCheckpoint -PathType Leaf) {
                $arguments += @('--resume-checkpoint', $lastCheckpoint)
                Write-Host ('[V21] ' + $Name + ' resumes from ' + $lastCheckpoint) -ForegroundColor Yellow
            }
            else {
                Move-PreCheckpointArtifact $Name $artifactRoot
            }
        }
    }
    Invoke-RadarPython $arguments
    Assert-CompletedVariant $Name
}

function Invoke-TrainAll {
    foreach ($name in $variants) {
        Invoke-TrainVariant $name
    }
}

function Invoke-Evaluate([string]$RequestedVariant) {
    Assert-Readiness
    Assert-CompletedVariant 'baseline'
    $arguments = @(
        '-m', 'report.known_gate_v21',
        '--project-root', $projectRoot
    )
    if ($RequestedVariant -eq 'all') {
        foreach ($name in @('v21a', 'v21b', 'v21c', 'v21d')) {
            Assert-CompletedVariant $name
        }
        $arguments += @('--variant', 'all')
        $reportPath = Join-Path $projectRoot 'src-v3\processed\report\RADAR-LN3-V21-known-gate-feature-validation.json'
    }
    else {
        Assert-CompletedVariant $RequestedVariant
        $arguments += @('--variant', $RequestedVariant)
        $version = Get-VariantVersion $RequestedVariant
        $reportPath = Join-Path $projectRoot (
            'src-v3\processed\report\RADAR-LN3-' + $version + '-known-gate-paired-validation.json'
        )
    }
    Invoke-RadarPython $arguments
    Assert-File $reportPath 'V21 KnownGate validation report'
    $report = Get-Content -LiteralPath $reportPath -Raw -Encoding UTF8 | ConvertFrom-Json
    if (
        $report.test -ne $notExecuted -or
        $report.OOF -ne $notExecuted -or
        $report.full_cascade -ne $notExecuted -or
        $report.formal_rejection -ne $notExecuted -or
        $report.stable -ne $notExecuted -or
        $report.P95 -ne $notExecuted -or
        $report.deployment_eligible -ne $false
    ) {
        throw ('V21 report violates the local/test-sealed contract: ' + $reportPath)
    }
}

try {
    Assert-File $python 'Project Python interpreter'
    Write-Host ('[V21] project=' + $projectRoot) -ForegroundColor Green
    Write-Host ('[V21] stage=' + $Stage + '; variant=' + $Variant + '; task=known_gate; seed=42') -ForegroundColor Green
    Write-Host (
        '[V21] Every run is a full 20-epoch from-initialization training run. ' +
        'Unknown is excluded from classes, optimization, normalization and checkpoint selection.'
    ) -ForegroundColor Green
    switch ($Stage) {
        'prepare' { Invoke-Prepare }
        'smoke' { Invoke-Smoke }
        'baseline' { Invoke-TrainVariant 'baseline' }
        'v21a' { Invoke-TrainVariant 'v21a' }
        'v21b' { Invoke-TrainVariant 'v21b' }
        'v21c' { Invoke-TrainVariant 'v21c' }
        'v21d' { Invoke-TrainVariant 'v21d' }
        'train-all' { Invoke-TrainAll }
        'evaluate' { Invoke-Evaluate $Variant }
        'all' {
            Invoke-Prepare
            Invoke-TrainAll
            Invoke-Evaluate 'all'
        }
    }
    Write-Host (
        '[V21] requested stage completed. OOF, test, full cascade, formal rejection, stable, P95 and deployment remain unexecuted.'
    ) -ForegroundColor Green
}
catch {
    Write-Error ('V21 KnownGate feature pipeline stopped: ' + $_.Exception.Message)
    exit 1
}
