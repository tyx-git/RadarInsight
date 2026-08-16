[CmdletBinding()]
param(
    [ValidateSet('prepare', 'train', 'evaluate', 'all')]
    [string]$Stage = 'all',
    [string]$ProjectRoot = ''
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$srcRoot = Split-Path -Parent $scriptRoot
$defaultProjectRoot = Split-Path -Parent $srcRoot
$projectPath = [System.IO.Path]::GetFullPath(
    $(if ([string]::IsNullOrWhiteSpace($ProjectRoot)) { $defaultProjectRoot } else { $ProjectRoot })
)
$python = Join-Path $projectPath '.venv\Scripts\python.exe'
$configRoot = Join-Path $projectPath 'config\RADAR-LN3'
$namePrefix = 'RADAR-LN3-V18r-cq08-liquid-dbb-specialist-rd-physical4-micro-doppler8-stable-density'
$controlConfig = Join-Path $configRoot ($namePrefix + '-control-reference-seed42-config.json')
$treatmentConfig = Join-Path $configRoot ($namePrefix + '-treatment-seed42-config.json')
$controlArtifact = Join-Path $projectPath ('artifacts\' + $namePrefix + '-control-reference-seed42-formal')
$treatmentArtifact = Join-Path $projectPath ('artifacts\' + $namePrefix + '-treatment-seed42-formal')
$sourceArtifact = Join-Path $projectPath (
    'artifacts\RADAR-LN3-V18-cq08-liquid-dbb-specialist-rd-physical4-' +
    'broadband-multiscale-micro-doppler8-stable-treatment-seed42-formal'
)
$sourceSummary = Join-Path $sourceArtifact 'training-summary.json'
$readinessPath = Join-Path $projectPath (
    'src-v3\processed\report\RADAR-LN3-V18r-stable8-density-weighted-training-readiness.json'
)
$pairedReportPath = Join-Path $projectPath (
    'src-v3\processed\report\RADAR-LN3-V18r-stable8-density-weighted-paired-validation.json'
)
$readinessSchema = 'RADAR-LN3-V18r-stable8-density-weighted-training-readiness-v1'
$pairedProtocol = 'v18r-dbb-stable8-bounded-observation-control-versus-density-weighted-treatment-v1'
$notExecuted = -join ([char[]](0x672A, 0x6267, 0x884C))

function Invoke-RadarPython {
    param([Parameter(Mandatory = $true)][string[]]$Arguments)

    $display = $python + ' ' + ($Arguments -join ' ')
    Write-Host ('> ' + $display) -ForegroundColor Cyan
    $exitCode = 0
    Push-Location $srcRoot
    try {
        & $python @Arguments
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

function Get-FileHashLower {
    param([Parameter(Mandatory = $true)][string]$Path)
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Move-PreCheckpointArtifactAside {
    $resolvedArtifact = [System.IO.Path]::GetFullPath($treatmentArtifact)
    $resolvedArtifactsRoot = [System.IO.Path]::GetFullPath(
        (Join-Path $projectPath 'artifacts')
    ).TrimEnd([System.IO.Path]::DirectorySeparatorChar) +
        [System.IO.Path]::DirectorySeparatorChar
    if (-not $resolvedArtifact.StartsWith(
        $resolvedArtifactsRoot,
        [System.StringComparison]::OrdinalIgnoreCase
    )) {
        throw ('Refusing to archive an artifact outside the project artifacts root: ' + $resolvedArtifact)
    }
    if (-not (Test-NonEmptyDirectory $resolvedArtifact)) {
        return $null
    }
    $lastCheckpoint = Join-Path $resolvedArtifact 'last_model.pt'
    if (Test-Path -LiteralPath $lastCheckpoint -PathType Leaf) {
        return $null
    }
    $epochDirectory = Join-Path $resolvedArtifact 'epochs'
    $epochCheckpoints = @()
    if (Test-Path -LiteralPath $epochDirectory -PathType Container) {
        $epochCheckpoints = @(
            Get-ChildItem -LiteralPath $epochDirectory -Filter '*.pt' -File
        )
    }
    if ($epochCheckpoints.Count -gt 0) {
        throw (
            'V18r artifact has epoch checkpoints but no last_model.pt; manual inspection is required: ' +
            $resolvedArtifact
        )
    }
    $archivePath = $resolvedArtifact + '-interrupted-before-first-checkpoint-' +
        (Get-Date -Format 'yyyyMMdd-HHmmss')
    if (Test-Path -LiteralPath $archivePath) {
        throw ('V18r pre-checkpoint archive destination already exists: ' + $archivePath)
    }
    Move-Item -LiteralPath $resolvedArtifact -Destination $archivePath
    return $archivePath
}

function Assert-SourceControl {
    Assert-File $sourceSummary 'Frozen V18 stable8 source summary'
    $summary = Read-Utf8Json $sourceSummary
    if (
        $summary.status -ne 'completed' -or
        $summary.task_name -ne 'dbb_specialist' -or
        $summary.stop_reason -ne 'epoch_limit' -or
        [int]$summary.best_epoch -ne 16 -or
        @($summary.history).Count -ne 20 -or
        $summary.test_evaluation -ne 'not_executed' -or
        $summary.full_cascade_evaluation -ne 'not_executed' -or
        $summary.oof_rejection_calibration -ne 'not_executed' -or
        $summary.stable_evaluation -ne 'not_executed' -or
        $summary.p95_latency -ne 'not_executed' -or
        $summary.deployment_eligible -ne $false
    ) {
        throw 'Frozen V18 stable8 source is incomplete or outside sealed scope.'
    }
    Assert-File (Join-Path $sourceArtifact 'best_model.pt') 'Frozen V18 best checkpoint'
    Assert-File (Join-Path $sourceArtifact 'last_model.pt') 'Frozen V18 last checkpoint'
    for ($epoch = 1; $epoch -le 20; $epoch++) {
        Assert-File (
            (Join-Path $sourceArtifact ('epochs\epoch-' + $epoch.ToString('000') + '.pt'))
        ) 'Frozen V18 epoch checkpoint'
    }
}

function Assert-Prepared {
    Assert-SourceControl
    Assert-File $controlConfig 'V18r Control-reference config'
    Assert-File $treatmentConfig 'V18r Treatment config'
    Assert-File $readinessPath 'V18r train-only readiness'
    if (Test-NonEmptyDirectory $controlArtifact) {
        throw ('V18r Control-reference artifact must remain empty: ' + $controlArtifact)
    }
    $readiness = Read-Utf8Json $readinessPath
    $failedChecks = @(
        $readiness.checks.PSObject.Properties |
            Where-Object { $_.Value -ne $true } |
            ForEach-Object { $_.Name }
    )
    if (
        $readiness.status -ne 'passed' -or
        $readiness.schema_version -ne $readinessSchema -or
        $readiness.paired_protocol -ne $pairedProtocol -or
        $readiness.validation_used_for_readiness_gate -ne $false -or
        [int]$readiness.test_payload_access_count -ne 0 -or
        $readiness.test -ne $notExecuted -or
        $readiness.deployment_eligible -ne $false -or
        $failedChecks.Count -ne 0
    ) {
        throw (
            'V18r readiness is stale, failed or unsealed; failed checks=' +
            ($failedChecks -join ',')
        )
    }
    foreach ($arm in @('control', 'treatment')) {
        $path = if ($arm -eq 'control') { $controlConfig } else { $treatmentConfig }
        $binding = $readiness.paired_configs.$arm
        if (
            [System.IO.Path]::GetFullPath([string]$binding.path) -ne
                [System.IO.Path]::GetFullPath($path) -or
            [string]$binding.sha256 -ne (Get-FileHashLower $path)
        ) {
            throw ('V18r readiness does not bind the current ' + $arm + ' config.')
        }
    }
}

function Assert-CompletedTreatment {
    $summaryPath = Join-Path $treatmentArtifact 'training-summary.json'
    Assert-File $summaryPath 'V18r Treatment summary'
    $summary = Read-Utf8Json $summaryPath
    $binding = $summary.v18r_training_readiness
    if (
        $summary.status -ne 'completed' -or
        $summary.task_name -ne 'dbb_specialist' -or
        $summary.stop_reason -ne 'epoch_limit' -or
        @($summary.history).Count -ne 20 -or
        $summary.test_evaluation -ne 'not_executed' -or
        $summary.full_cascade_evaluation -ne 'not_executed' -or
        $summary.oof_rejection_calibration -ne 'not_executed' -or
        $summary.stable_evaluation -ne 'not_executed' -or
        $summary.p95_latency -ne 'not_executed' -or
        $summary.deployment_eligible -ne $false -or
        $binding.status -ne 'passed' -or
        $binding.schema_version -ne $readinessSchema -or
        $binding.paired_protocol -ne $pairedProtocol
    ) {
        throw 'V18r Treatment is incomplete or violates the sealed local-task contract.'
    }
    Assert-File ([string]$binding.path) 'V18r summary readiness binding'
    if ([string]$binding.sha256 -ne (Get-FileHashLower ([string]$binding.path))) {
        throw 'V18r summary readiness SHA binding is invalid.'
    }
    Assert-File (Join-Path $treatmentArtifact 'best_model.pt') 'V18r best checkpoint'
    Assert-File (Join-Path $treatmentArtifact 'last_model.pt') 'V18r last checkpoint'
    for ($epoch = 1; $epoch -le 20; $epoch++) {
        Assert-File (
            (Join-Path $treatmentArtifact ('epochs\epoch-' + $epoch.ToString('000') + '.pt'))
        ) 'V18r epoch checkpoint'
    }
    $source = Read-Utf8Json $sourceSummary
    for ($index = 0; $index -lt 20; $index++) {
        if (
            [string]$source.history[$index].parent_order_sha256 -ne
            [string]$summary.history[$index].parent_order_sha256
        ) {
            throw ('V18r parent order differs from frozen V18 at epoch ' + ($index + 1))
        }
    }
    if (
        [string]$source.paired_initialization.initial_logits_sha256 -ne
        [string]$summary.paired_initialization.initial_logits_sha256
    ) {
        throw 'V18r initial logits do not reproduce the frozen V18 stable8 source.'
    }
}

function Invoke-Prepare {
    Write-Host '[V18r] stage=prepare; frozen V18 stable8 Control; density Treatment only' -ForegroundColor Green
    Invoke-RadarPython @(
        '-m', 'py_compile',
        'radar_ln3\task_models.py',
        'train\tasks.py',
        'build\stable8_density_v18r.py',
        'report\stable8_density_v18r_readiness.py',
        'report\stable8_density_v18r_paired.py',
        'smoke\stable8_density_v18r_contract_test.py'
    )
    Assert-SourceControl
    if (-not (Test-NonEmptyDirectory $treatmentArtifact)) {
        Invoke-RadarPython @(
            '-m', 'build.stable8_density_v18r',
            '--project-root', $projectPath
        )
        Invoke-RadarPython @(
            '-m', 'report.stable8_density_v18r_readiness',
            '--project-root', $projectPath
        )
    }
    else {
        Write-Host '[V18r] Treatment artifact is non-empty; immutable prepare generation is skipped.' -ForegroundColor Yellow
    }
    Assert-Prepared
    Invoke-RadarPython @(
        '-m', 'smoke.static_isolation_check',
        '--root', $srcRoot
    )
    Invoke-RadarPython @(
        '-m', 'smoke.stable8_density_v18r_contract_test',
        '--project-root', $projectPath
    )
}

function Invoke-Train {
    Write-Host '[V18r] stage=train; density Treatment only; seed=42' -ForegroundColor Green
    Assert-Prepared
    $summaryPath = Join-Path $treatmentArtifact 'training-summary.json'
    $lastCheckpoint = Join-Path $treatmentArtifact 'last_model.pt'
    if (Test-Path -LiteralPath $summaryPath -PathType Leaf) {
        $existing = Read-Utf8Json $summaryPath
        if ($existing.status -eq 'completed') {
            Assert-CompletedTreatment
            Write-Host '[V18r] Treatment already completed; preserving the formal artifact.' -ForegroundColor Yellow
            return
        }
    }
    $arguments = @(
        '-m', 'train.tasks',
        '--project-root', $projectPath,
        '--config', $treatmentConfig
    )
    if (Test-NonEmptyDirectory $treatmentArtifact) {
        if (Test-Path -LiteralPath $lastCheckpoint -PathType Leaf) {
            $arguments += @('--resume-checkpoint', $lastCheckpoint)
            Write-Host ('[V18r] Treatment resumes from ' + $lastCheckpoint) -ForegroundColor Yellow
        }
        else {
            $archivePath = Move-PreCheckpointArtifactAside
            if ([string]::IsNullOrWhiteSpace([string]$archivePath)) {
                throw 'V18r non-empty artifact has no resumable checkpoint and was not archived.'
            }
            Write-Host (
                '[V18r] Archived the pre-checkpoint interrupted run and will restart from ' +
                'the frozen V18 initialization: ' + $archivePath
            ) -ForegroundColor Yellow
        }
    }
    Invoke-RadarPython $arguments
    Assert-CompletedTreatment
}

function Invoke-Evaluate {
    Write-Host '[V18r] stage=evaluate; frozen V18 stable8 versus V18r density' -ForegroundColor Green
    Assert-Prepared
    Assert-CompletedTreatment
    Invoke-RadarPython @(
        '-m', 'report.stable8_density_v18r_paired',
        '--project-root', $projectPath
    )
    Assert-File $pairedReportPath 'V18r paired validation report'
    $report = Read-Utf8Json $pairedReportPath
    if (
        $report.test -ne $notExecuted -or
        $report.OOF -ne $notExecuted -or
        $report.full_cascade -ne $notExecuted -or
        $report.formal_rejection -ne $notExecuted -or
        $report.stable -ne $notExecuted -or
        $report.P95 -ne $notExecuted -or
        $report.deployment_eligible -ne $false -or
        $report.control_retrained -ne $false
    ) {
        throw 'V18r report violates frozen-Control or sealed-test scope.'
    }
}

try {
    Assert-File $python 'Project Python interpreter'
    Write-Host ('[V18r] project=' + $projectPath) -ForegroundColor Green
    Write-Host ('[V18r] requested stage=' + $Stage) -ForegroundColor Green
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
    Write-Host '[V18r] requested stages completed. OOF, test, cascade, rejection, stable, P95 and deployment remain unexecuted.' -ForegroundColor Green
}
catch {
    [Console]::Error.WriteLine('[V18r] stable8-density pipeline stopped: ' + $_.Exception.Message)
    exit 1
}
