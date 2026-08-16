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
$configPath = Join-Path $projectPath (
    'config\RADAR-LN3\RADAR-LN3-V19r-cq08-liquid-dbb-specialist-rd-physical4-' +
    'micro-doppler8-stable-density-joint-late-residual-seed42-config.json'
)
$readinessPath = Join-Path $projectPath (
    'src-v3\processed\report\RADAR-LN3-V19r-stable8-density-' +
    'joint-late-residual-training-readiness.json'
)
$reportPath = Join-Path $projectPath (
    'src-v3\processed\report\RADAR-LN3-V19r-stable8-density-' +
    'joint-late-residual-validation.json'
)
$artifactPath = Join-Path $projectPath (
    'artifacts\RADAR-LN3-V19r-cq08-liquid-dbb-specialist-rd-physical4-' +
    'micro-doppler8-stable-density-joint-late-residual-seed42-formal'
)
$summaryPath = Join-Path $artifactPath 'training-summary.json'
$lastCheckpointPath = Join-Path $artifactPath 'last_model.pt'
$readinessSchema = 'RADAR-LN3-V19r-stable8-density-joint-late-residual-readiness-v1'
$pairedProtocol = 'v19r-dbb-v18r-full-joint-plus-stable8-late-residual-v1'
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
    $resolvedArtifact = [System.IO.Path]::GetFullPath($artifactPath)
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
    if (Test-Path -LiteralPath $lastCheckpointPath -PathType Leaf) {
        return $null
    }
    $epochDirectory = Join-Path $resolvedArtifact 'epochs'
    $epochCheckpoints = @()
    if (Test-Path -LiteralPath $epochDirectory -PathType Container) {
        $epochCheckpoints = @(Get-ChildItem -LiteralPath $epochDirectory -Filter '*.pt' -File)
    }
    if ($epochCheckpoints.Count -gt 0) {
        throw (
            'V19r artifact has epoch checkpoints but no last_model.pt; manual inspection is required: ' +
            $resolvedArtifact
        )
    }
    $archivePath = $resolvedArtifact + '-interrupted-before-first-checkpoint-' +
        (Get-Date -Format 'yyyyMMdd-HHmmss')
    if (Test-Path -LiteralPath $archivePath) {
        throw ('V19r pre-checkpoint archive destination already exists: ' + $archivePath)
    }
    Move-Item -LiteralPath $resolvedArtifact -Destination $archivePath
    return $archivePath
}

function Assert-Prepared {
    Assert-File $configPath 'V19r config'
    Assert-File $readinessPath 'V19r readiness'
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
        [int]$readiness.backbone_trainable_parameter_count -le 24 -or
        [int]$readiness.residual_trainable_parameter_count -ne 24 -or
        [int]$readiness.total_trainable_parameter_count -ne
            ([int]$readiness.backbone_trainable_parameter_count + 24) -or
        $readiness.validation_used_for_readiness_gate -ne $false -or
        [int]$readiness.test_payload_access_count -ne 0 -or
        $readiness.test -ne $notExecuted -or
        $readiness.deployment_eligible -ne $false -or
        $failedChecks.Count -ne 0 -or
        [System.IO.Path]::GetFullPath([string]$readiness.config_path) -ne
            [System.IO.Path]::GetFullPath($configPath) -or
        [string]$readiness.config_sha256 -ne (Get-FileHashLower $configPath)
    ) {
        throw (
            'V19r readiness is stale, failed or unsealed; failed checks=' +
            ($failedChecks -join ',')
        )
    }
}

function Assert-Completed {
    Assert-Prepared
    Assert-File $summaryPath 'V19r training summary'
    $summary = Read-Utf8Json $summaryPath
    if (
        $summary.status -ne 'completed' -or
        $summary.config_name -ne 'RADAR-LN3-V19r-cq08-liquid-dbb-specialist-rd-physical4-micro-doppler8-stable-density-joint-late-residual-seed42' -or
        $summary.paired_protocol -ne $pairedProtocol -or
        $summary.task_name -ne 'dbb_specialist' -or
        $summary.stop_reason -ne 'epoch_limit' -or
        @($summary.history).Count -ne 20 -or
        [int]$summary.backbone_trainable_parameter_count -le 24 -or
        [int]$summary.residual_trainable_parameter_count -ne 24 -or
        [int]$summary.total_trainable_parameter_count -ne
            ([int]$summary.backbone_trainable_parameter_count + 24) -or
        $summary.test_evaluation -ne 'not_executed' -or
        $summary.full_cascade_evaluation -ne 'not_executed' -or
        $summary.oof_rejection_calibration -ne 'not_executed' -or
        $summary.stable_evaluation -ne 'not_executed' -or
        $summary.p95_latency -ne 'not_executed' -or
        $summary.deployment_eligible -ne $false
    ) {
        throw 'V19r training summary is incomplete or outside the sealed local-task scope.'
    }
    Assert-File (Join-Path $artifactPath 'best_model.pt') 'V19r best checkpoint'
    Assert-File $lastCheckpointPath 'V19r last checkpoint'
    for ($epoch = 1; $epoch -le 20; $epoch++) {
        Assert-File (
            (Join-Path $artifactPath ('epochs\epoch-' + $epoch.ToString('000') + '.pt'))
        ) 'V19r epoch checkpoint'
    }
}

function Invoke-Prepare {
    Write-Host '[V19r] stage=prepare; complete V18r backbone plus zero-init residual, all trainable' -ForegroundColor Green
    Invoke-RadarPython @(
        '-m', 'py_compile',
        'radar_ln3\micro_doppler_joint_residual_v19r.py',
        'build\micro_doppler_joint_residual_v19r.py',
        'train\micro_doppler_joint_residual_v19r.py',
        'report\micro_doppler_joint_residual_v19r.py',
        'smoke\micro_doppler_joint_residual_v19r_contract_test.py'
    )
    foreach ($packageName in @('build', 'radar_ln3', 'report', 'smoke', 'tests', 'train', 'validate')) {
        $packageRoot = Join-Path $srcRoot $packageName
        if (Test-Path -LiteralPath $packageRoot -PathType Container) {
            Invoke-RadarPython @(
                '-m', 'smoke.static_isolation_check',
                '--root', $packageRoot
            )
        }
    }
    Invoke-RadarPython @(
        '-m', 'smoke.micro_doppler_joint_residual_v19r_contract_test',
        '--project-root', $projectPath
    )
    if (Test-NonEmptyDirectory $artifactPath) {
        Write-Host '[V19r] formal artifact is non-empty; immutable prepare generation is skipped.' -ForegroundColor Yellow
    }
    else {
        Invoke-RadarPython @(
            '-m', 'build.micro_doppler_joint_residual_v19r',
            '--project-root', $projectPath
        )
    }
    Assert-Prepared
    Invoke-RadarPython @(
        '-m', 'smoke.micro_doppler_joint_residual_v19r_contract_test',
        '--project-root', $projectPath,
        '--prepared'
    )
}

function Invoke-Train {
    Write-Host '[V19r] stage=train; full from-scratch backbone and residual joint training' -ForegroundColor Green
    Assert-Prepared
    if (Test-Path -LiteralPath $summaryPath -PathType Leaf) {
        $existing = Read-Utf8Json $summaryPath
        if ($existing.status -eq 'completed') {
            Assert-Completed
            Write-Host '[V19r] training already completed; preserving the formal artifact.' -ForegroundColor Yellow
            return
        }
    }
    $arguments = @(
        '-m', 'train.micro_doppler_joint_residual_v19r',
        '--project-root', $projectPath,
        '--config', $configPath
    )
    if (Test-NonEmptyDirectory $artifactPath) {
        if (Test-Path -LiteralPath $lastCheckpointPath -PathType Leaf) {
            $arguments += @('--resume-checkpoint', $lastCheckpointPath)
            Write-Host ('[V19r] resuming from ' + $lastCheckpointPath) -ForegroundColor Yellow
        }
        else {
            $archivePath = Move-PreCheckpointArtifactAside
            if ([string]::IsNullOrWhiteSpace([string]$archivePath)) {
                throw 'V19r non-empty artifact has no resumable checkpoint and was not archived.'
            }
            Write-Host (
                '[V19r] archived the pre-checkpoint interrupted run and will restart from the V18r initial state: ' +
                $archivePath
            ) -ForegroundColor Yellow
        }
    }
    Invoke-RadarPython $arguments
    Assert-Completed
}

function Invoke-Evaluate {
    Write-Host '[V19r] stage=evaluate; V18r reference versus full joint V19r treatment' -ForegroundColor Green
    Assert-Completed
    Invoke-RadarPython @(
        '-m', 'report.micro_doppler_joint_residual_v19r',
        '--project-root', $projectPath,
        '--config', $configPath
    )
    Assert-File $reportPath 'V19r validation report'
    $report = Read-Utf8Json $reportPath
    if (
        $report.test -ne $notExecuted -or
        $report.OOF -ne $notExecuted -or
        $report.full_cascade -ne $notExecuted -or
        $report.formal_rejection -ne $notExecuted -or
        $report.stable -ne $notExecuted -or
        $report.P95 -ne $notExecuted -or
        $report.deployment_eligible -ne $false -or
        $report.control.retrained -ne $false -or
        $report.treatment.trained_from_scratch -ne $true -or
        [int]$report.treatment.backbone_trainable_parameter_count -le 24 -or
        [int]$report.treatment.residual_trainable_parameter_count -ne 24
    ) {
        throw 'V19r report violates frozen-baseline or sealed-test scope.'
    }
}

try {
    Assert-File $python 'Project Python interpreter'
    Write-Host ('[V19r] project=' + $projectPath) -ForegroundColor Green
    Write-Host ('[V19r] requested stage=' + $Stage) -ForegroundColor Green
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
    Write-Host '[V19r] requested stages completed. OOF, test, cascade, rejection, stable, P95 and deployment remain unexecuted.' -ForegroundColor Green
}
catch {
    [Console]::Error.WriteLine('[V19r] full joint late-residual pipeline stopped: ' + $_.Exception.Message)
    exit 1
}
