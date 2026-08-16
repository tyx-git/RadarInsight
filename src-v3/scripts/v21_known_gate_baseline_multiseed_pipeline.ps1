[CmdletBinding()]
param(
    [ValidateSet('prepare', 'smoke', 'seed43', 'seed44', 'train', 'evaluate', 'all')]
    [string]$Stage = 'all'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$srcRoot = Split-Path -Parent $scriptRoot
$projectRoot = Split-Path -Parent $srcRoot
$python = Join-Path $projectRoot '.venv\Scripts\python.exe'
$configRoot = Join-Path $projectRoot 'config\RADAR-LN3'
$readinessPath = Join-Path $projectRoot 'src-v3\processed\report\RADAR-LN3-V21-known-gate-baseline-multiseed-training-readiness.json'
$reportPath = Join-Path $projectRoot 'src-v3\processed\report\RADAR-LN3-V21-known-gate-baseline-multiseed-validation.json'
$selectionPath = Join-Path $projectRoot 'data\processed\RADAR-LN3-V21-known-gate-baseline-development-selection.json'
$readinessSchema = 'RADAR-LN3-V21-known-gate-baseline-multiseed-training-readiness-v1'
$notExecuted = -join ([char[]](0x672A, 0x6267, 0x884C))
$seeds = @(42, 43, 44)
$newSeeds = @(43, 44)

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

function Get-ConfigPath([int]$Seed) {
    return Join-Path $configRoot (
        'RADAR-LN3-V21-cq08-liquid-known-gate-baseline-seed' + $Seed + '-config.json'
    )
}

function Get-ArtifactRoot([int]$Seed) {
    return Join-Path $projectRoot (
        'artifacts\RADAR-LN3-V21-cq08-liquid-known-gate-baseline-seed' + $Seed + '-formal'
    )
}

function Assert-MultiseedReadiness {
    Assert-File $readinessPath 'V21 Baseline multiseed readiness'
    $readiness = Get-Content -LiteralPath $readinessPath -Raw -Encoding UTF8 | ConvertFrom-Json
    if (
        $readiness.status -ne 'passed' -or
        $readiness.schema_version -ne $readinessSchema -or
        $readiness.test -ne $notExecuted -or
        $readiness.test_payload_access_count -ne 0 -or
        $readiness.deployment_eligible -ne $false
    ) {
        throw ('V21 Baseline multiseed readiness violates the sealed contract: ' + $readinessPath)
    }
    $observedSeeds = @($readiness.seeds | ForEach-Object { [int]$_ })
    if (($observedSeeds -join ',') -ne ($seeds -join ',')) {
        throw ('V21 Baseline multiseed seed set differs: ' + ($observedSeeds -join ','))
    }
    $failed = @(
        $readiness.checks.PSObject.Properties |
            Where-Object { $_.Value -ne $true } |
            ForEach-Object { $_.Name }
    )
    if ($failed.Count -gt 0) {
        throw ('V21 Baseline multiseed readiness contains failed checks: ' + ($failed -join ','))
    }
    foreach ($seed in $seeds) {
        $configPath = Get-ConfigPath $seed
        Assert-File $configPath ('V21 Baseline seed' + $seed + ' config')
        $seedName = [string]$seed
        $bindingProperty = $readiness.configs.PSObject.Properties[$seedName]
        $binding = if ($null -eq $bindingProperty) { $null } else { $bindingProperty.Value }
        $actualHash = (Get-FileHash -LiteralPath $configPath -Algorithm SHA256).Hash.ToLowerInvariant()
        if (
            $null -eq $binding -or
            [System.IO.Path]::GetFullPath([string]$binding.path) -ne [System.IO.Path]::GetFullPath($configPath) -or
            $binding.sha256 -ne $actualHash
        ) {
            throw ('V21 Baseline readiness does not bind seed' + $seed + ' config: ' + $configPath)
        }
    }
}

function Assert-CompletedSeed([int]$Seed) {
    $artifactRoot = Get-ArtifactRoot $Seed
    $summaryPath = Join-Path $artifactRoot 'training-summary.json'
    Assert-File $summaryPath ('V21 Baseline seed' + $Seed + ' summary')
    $summary = Get-Content -LiteralPath $summaryPath -Raw -Encoding UTF8 | ConvertFrom-Json
    $observedSeed = if ($summary.PSObject.Properties.Name -contains 'seed') {
        [int]$summary.seed
    }
    else {
        [int]$summary.dataloader_contract.seed
    }
    if (
        $summary.status -ne 'completed' -or
        $summary.task_name -ne 'known_gate' -or
        $summary.variant -ne 'baseline' -or
        $observedSeed -ne $Seed -or
        $summary.train_parent_count -ne 986 -or
        $summary.validation_parent_count -ne 232 -or
        $summary.validation_clutter_support -ne 9 -or
        @($summary.history).Count -ne 20 -or
        $summary.full_parameter_training -ne $true -or
        $summary.all_model_parameters_trainable -ne $true -or
        $summary.model_state_changed_from_initialization -ne $true -or
        $summary.test_evaluation -ne 'not_executed' -or
        $summary.oof_threshold_calibration -ne 'not_executed' -or
        $summary.full_cascade_evaluation -ne 'not_executed' -or
        $summary.formal_rejection -ne 'not_executed' -or
        $summary.stable_evaluation -ne 'not_executed' -or
        $summary.p95_latency -ne 'not_executed' -or
        $summary.deployment_eligible -ne $false
    ) {
        throw ('V21 Baseline seed' + $Seed + ' is not a completed full run: ' + $summaryPath)
    }
    Assert-File (Join-Path $artifactRoot 'best_model.pt') ('V21 Baseline seed' + $Seed + ' best checkpoint')
    Assert-File (Join-Path $artifactRoot 'last_model.pt') ('V21 Baseline seed' + $Seed + ' last checkpoint')
    Assert-File (Join-Path $artifactRoot 'sample_detail_information.json') ('V21 Baseline seed' + $Seed + ' sample detail')
    for ($epoch = 1; $epoch -le 20; $epoch++) {
        $epochName = 'epoch-' + $epoch.ToString('000') + '.pt'
        Assert-File (Join-Path $artifactRoot ('epochs\' + $epochName)) ('V21 Baseline seed' + $Seed + ' epoch checkpoint')
    }
}

function Invoke-Smoke {
    Invoke-RadarPython @(
        '-m', 'smoke.known_gate_v21_multiseed_contract_test',
        '--project-root', $projectRoot
    )
    Invoke-RadarPython @(
        '-m', 'smoke.static_isolation_check',
        '--root', $srcRoot
    )
}

function Invoke-Prepare {
    Invoke-RadarPython @(
        '-m', 'build.known_gate_v21_multiseed',
        '--project-root', $projectRoot
    )
    Invoke-Smoke
    Assert-MultiseedReadiness
    Assert-CompletedSeed 42
}

function Move-PreCheckpointArtifact([int]$Seed, [string]$ArtifactRoot) {
    $checkpointFiles = @(
        Get-ChildItem -LiteralPath $ArtifactRoot -Recurse -File -Filter '*.pt' -ErrorAction SilentlyContinue
    )
    if ($checkpointFiles.Count -gt 0) {
        throw ('V21 Baseline seed' + $Seed + ' has checkpoint files but no resumable last_model.pt: ' + $ArtifactRoot)
    }
    $archive = $ArtifactRoot + '.interrupted-before-checkpoint-' + (Get-Date -Format 'yyyyMMdd-HHmmss')
    if (Test-Path -LiteralPath $archive) {
        throw ('V21 Baseline pre-checkpoint archive already exists: ' + $archive)
    }
    Move-Item -LiteralPath $ArtifactRoot -Destination $archive
    Write-Host ('[V21 multiseed] archived pre-checkpoint artifact: ' + $archive) -ForegroundColor Yellow
}

function Invoke-TrainSeed([int]$Seed) {
    Assert-MultiseedReadiness
    if ($Seed -eq 42) {
        Assert-CompletedSeed 42
        Write-Host '[V21 multiseed] seed42 already completed and is reused without retraining.' -ForegroundColor Yellow
        return
    }
    $configPath = Get-ConfigPath $Seed
    $artifactRoot = Get-ArtifactRoot $Seed
    $summaryPath = Join-Path $artifactRoot 'training-summary.json'
    if (Test-Path -LiteralPath $summaryPath -PathType Leaf) {
        $summary = Get-Content -LiteralPath $summaryPath -Raw -Encoding UTF8 | ConvertFrom-Json
        if ($summary.status -eq 'completed') {
            Assert-CompletedSeed $Seed
            Write-Host ('[V21 multiseed] seed' + $Seed + ' already completed; preserving artifact.') -ForegroundColor Yellow
            return
        }
        if ($summary.status -ne 'failed') {
            throw ('V21 Baseline seed' + $Seed + ' has unsupported summary status: ' + $summaryPath)
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
                Write-Host ('[V21 multiseed] seed' + $Seed + ' resumes from ' + $lastCheckpoint) -ForegroundColor Yellow
            }
            else {
                Move-PreCheckpointArtifact $Seed $artifactRoot
            }
        }
    }
    Invoke-RadarPython $arguments
    Assert-CompletedSeed $Seed
}

function Invoke-TrainNewSeeds {
    foreach ($seed in $newSeeds) {
        Invoke-TrainSeed $seed
    }
}

function Invoke-Evaluate {
    Assert-MultiseedReadiness
    foreach ($seed in $seeds) {
        Assert-CompletedSeed $seed
    }
    Invoke-RadarPython @(
        '-m', 'report.known_gate_v21_multiseed',
        '--project-root', $projectRoot
    )
    Assert-File $reportPath 'V21 Baseline multiseed validation report'
    Assert-File $selectionPath 'V21 Baseline development selection artifact'
    $report = Get-Content -LiteralPath $reportPath -Raw -Encoding UTF8 | ConvertFrom-Json
    $selection = Get-Content -LiteralPath $selectionPath -Raw -Encoding UTF8 | ConvertFrom-Json
    if (
        $report.status -ne 'completed' -or
        $report.test -ne $notExecuted -or
        $report.OOF -ne $notExecuted -or
        $report.deployment_eligible -ne $false -or
        $selection.status -ne 'selected_development_champion' -or
        $selection.selected_seed -notin $seeds -or
        $selection.test -ne $notExecuted -or
        $selection.deployment_eligible -ne $false
    ) {
        throw ('V21 Baseline multiseed report violates the development-only contract: ' + $reportPath)
    }
}

try {
    Assert-File $python 'Project Python interpreter'
    Write-Host ('[V21 multiseed] project=' + $projectRoot) -ForegroundColor Green
    Write-Host ('[V21 multiseed] stage=' + $Stage + '; seeds=42,43,44; seed42=reuse; seed43/44=full20epoch') -ForegroundColor Green
    switch ($Stage) {
        'prepare' { Invoke-Prepare }
        'smoke' { Invoke-Smoke }
        'seed43' { Invoke-TrainSeed 43 }
        'seed44' { Invoke-TrainSeed 44 }
        'train' { Invoke-TrainNewSeeds }
        'evaluate' { Invoke-Evaluate }
        'all' {
            Invoke-Prepare
            Invoke-TrainNewSeeds
            Invoke-Evaluate
        }
    }
    Write-Host '[V21 multiseed] requested stage completed. Selected model remains development-only; test/OOF/rejection/stable/P95 are unexecuted.' -ForegroundColor Green
}
catch {
    Write-Error ('V21 KnownGate Baseline multiseed pipeline stopped: ' + $_.Exception.Message)
    exit 1
}
