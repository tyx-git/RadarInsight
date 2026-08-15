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
$readinessPath = Join-Path $projectPath (
    'src-v3\processed\report\RADAR-LN3-V20-density-weighted-training-readiness.json'
)
$pairedReportPath = Join-Path $projectPath (
    'src-v3\processed\report\RADAR-LN3-V20-density-weighted-paired-validation.json'
)
$sharedControlArtifact = Join-Path $projectPath (
    'artifacts\RADAR-LN3-V20-cq08-liquid-dbb-specialist-zero-delta-skip-' +
    'control-seed42-formal'
)
$sharedControlSummary = Join-Path $sharedControlArtifact 'training-summary.json'
$notExecuted = -join ([char[]](0x672A, 0x6267, 0x884C))
$readinessSchema = 'RADAR-LN3-V20-density-weighted-training-readiness-v1'
$pairedProtocol = 'v20r-dbb-bounded-observation-control-versus-density-weighted-treatment-v1'
$sharedProtocol = 'v20-dbb-bounded-observation-control-versus-zero-delta-skip-treatment-v1'

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw ('Project Python interpreter is missing: ' + $python)
}
if (-not (Test-Path -LiteralPath $srcRoot -PathType Container)) {
    throw ('src-v3 directory is missing: ' + $srcRoot)
}

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

function Get-ConfigPath {
    param([Parameter(Mandatory = $true)][ValidateSet('control', 'treatment')][string]$Arm)

    return Join-Path $configRoot (
        'RADAR-LN3-V20r-cq08-liquid-dbb-specialist-density-weighted-' +
        $Arm + '-seed42-config.json'
    )
}

function Get-ArtifactPath {
    param([Parameter(Mandatory = $true)][ValidateSet('control', 'treatment')][string]$Arm)

    return Join-Path $projectPath (
        'artifacts\RADAR-LN3-V20r-cq08-liquid-dbb-specialist-density-weighted-' +
        $Arm + '-seed42-formal'
    )
}

function Assert-SharedControl {
    Assert-File $sharedControlSummary 'V20r shared V20 skip Control summary'
    $summary = Read-Utf8Json $sharedControlSummary
    $required = @(
        'status', 'task_name', 'stop_reason', 'v20_training_readiness',
        'test_evaluation', 'full_cascade_evaluation', 'oof_rejection_calibration',
        'stable_evaluation', 'p95_latency', 'deployment_eligible', 'history'
    )
    foreach ($fieldName in $required) {
        if (-not ($summary.PSObject.Properties.Name -contains $fieldName)) {
            throw ('Shared V20 Control summary is missing field ' + $fieldName)
        }
    }
    $binding = $summary.v20_training_readiness
    if (
        $summary.status -ne 'completed' -or
        $summary.task_name -ne 'dbb_specialist' -or
        $summary.stop_reason -ne 'epoch_limit' -or
        $summary.test_evaluation -ne 'not_executed' -or
        $summary.full_cascade_evaluation -ne 'not_executed' -or
        $summary.oof_rejection_calibration -ne 'not_executed' -or
        $summary.stable_evaluation -ne 'not_executed' -or
        $summary.p95_latency -ne 'not_executed' -or
        $summary.deployment_eligible -ne $false -or
        $binding.status -ne 'passed' -or
        $binding.paired_protocol -ne $sharedProtocol -or
        @($summary.history).Count -ne 20
    ) {
        throw 'V20r shared V20 skip Control is incomplete or outside sealed-test scope.'
    }
    Assert-File ([string]$binding.path) 'Shared V20 Control readiness binding'
    if ([string]$binding.sha256 -ne (Get-FileHashLower ([string]$binding.path))) {
        throw 'Shared V20 Control readiness SHA binding is invalid.'
    }
    Assert-File (Join-Path $sharedControlArtifact 'best_model.pt') 'Shared V20 Control best checkpoint'
    Assert-File (Join-Path $sharedControlArtifact 'last_model.pt') 'Shared V20 Control last checkpoint'
    for ($epoch = 1; $epoch -le 20; $epoch++) {
        Assert-File (
            Join-Path $sharedControlArtifact ('epochs\epoch-' + $epoch.ToString('000') + '.pt')
        ) 'Shared V20 Control epoch checkpoint'
    }
}

function Assert-Prepared {
    Assert-SharedControl
    if (Test-NonEmptyDirectory (Get-ArtifactPath 'control')) {
        throw (
            'V20r density Control must reuse the completed V20 Control and must not be retrained: ' +
            (Get-ArtifactPath 'control')
        )
    }
    Assert-File $readinessPath 'V20r density train-only readiness report'
    $readiness = Read-Utf8Json $readinessPath
    if (-not ($readiness.PSObject.Properties.Name -contains 'checks')) {
        throw ('V20r readiness has no checks object: ' + $readinessPath)
    }
    $failedChecks = @(
        $readiness.checks.PSObject.Properties |
            Where-Object { $_.Value -ne $true }
    )
    if ($failedChecks.Count -gt 0) {
        throw (
            'V20r readiness contains failed checks=' +
            (($failedChecks | ForEach-Object { $_.Name }) -join ',')
        )
    }
    if (
        $readiness.status -ne 'passed' -or
        $readiness.schema_version -ne $readinessSchema -or
        $readiness.paired_protocol -ne $pairedProtocol -or
        $readiness.validation_used_for_readiness_gate -ne $false -or
        $readiness.test -ne $notExecuted -or
        $readiness.test_payload_access_count -ne 0 -or
        $readiness.deployment_eligible -ne $false -or
        $readiness.shared_control_reuse.shared_control_retrained -ne $false -or
        $readiness.shared_control_reuse.summary_sha256 -ne (Get-FileHashLower $sharedControlSummary)
    ) {
        throw ('V20r readiness is not passed/shared-Control/test-sealed: ' + $readinessPath)
    }
    foreach ($arm in @('control', 'treatment')) {
        $configPath = Get-ConfigPath $arm
        Assert-File $configPath ('V20r ' + $arm + ' config')
        if (-not ($readiness.paired_configs.PSObject.Properties.Name -contains $arm)) {
            throw ('V20r readiness lacks the ' + $arm + ' config binding')
        }
        $binding = $readiness.paired_configs.$arm
        if (
            [System.IO.Path]::GetFullPath([string]$binding.path) -ne
                [System.IO.Path]::GetFullPath($configPath) -or
            [string]$binding.sha256 -ne (Get-FileHashLower $configPath)
        ) {
            throw ('V20r readiness does not bind the ' + $arm + ' config')
        }
    }
}

function Assert-CompletedTreatment {
    $artifactPath = Get-ArtifactPath 'treatment'
    $summaryPath = Join-Path $artifactPath 'training-summary.json'
    Assert-File $summaryPath 'V20r Treatment training summary'
    $summary = Read-Utf8Json $summaryPath
    $required = @(
        'status', 'performance_status', 'task_name', 'stop_reason',
        'v20_training_readiness', 'test_evaluation', 'full_cascade_evaluation',
        'oof_rejection_calibration', 'stable_evaluation', 'p95_latency',
        'deployment_eligible', 'history', 'best_epoch'
    )
    foreach ($fieldName in $required) {
        if (-not ($summary.PSObject.Properties.Name -contains $fieldName)) {
            throw ('V20r Treatment summary is missing field ' + $fieldName)
        }
    }
    $binding = $summary.v20_training_readiness
    if (
        $summary.status -ne 'completed' -or
        $summary.task_name -ne 'dbb_specialist' -or
        $summary.stop_reason -ne 'epoch_limit' -or
        $summary.test_evaluation -ne 'not_executed' -or
        $summary.full_cascade_evaluation -ne 'not_executed' -or
        $summary.oof_rejection_calibration -ne 'not_executed' -or
        $summary.stable_evaluation -ne 'not_executed' -or
        $summary.p95_latency -ne 'not_executed' -or
        $summary.deployment_eligible -ne $false -or
        $binding.status -ne 'passed' -or
        $binding.schema_version -ne $readinessSchema -or
        $binding.paired_protocol -ne $pairedProtocol -or
        @($summary.history).Count -ne 20
    ) {
        throw 'V20r Treatment is incomplete or violates the sealed local-task contract.'
    }
    Assert-File ([string]$binding.path) 'V20r Treatment readiness binding'
    if ([string]$binding.sha256 -ne (Get-FileHashLower ([string]$binding.path))) {
        throw 'V20r Treatment readiness SHA binding is invalid.'
    }
    Assert-File (Join-Path $artifactPath 'best_model.pt') 'V20r Treatment best checkpoint'
    Assert-File (Join-Path $artifactPath 'last_model.pt') 'V20r Treatment last checkpoint'
    for ($epoch = 1; $epoch -le 20; $epoch++) {
        Assert-File (
            Join-Path $artifactPath ('epochs\epoch-' + $epoch.ToString('000') + '.pt')
        ) 'V20r Treatment epoch checkpoint'
    }
}

function Invoke-Prepare {
    Write-Host '[V20r] stage=prepare; variant=density-weighted; shared Control is read-only' -ForegroundColor Green
    Invoke-RadarPython @(
        '-m', 'py_compile',
        'radar_ln3\model.py',
        'radar_ln3\config.py',
        'radar_ln3\streaming.py',
        'radar_ln3\cascade.py',
        'radar_ln3\task_models.py',
        'train\tasks.py',
        'build\liquid_dynamics_v20.py',
        'report\liquid_dynamics_v20_readiness.py',
        'report\liquid_dynamics_v20_paired.py',
        'smoke\liquid_dynamics_v20_contract_test.py'
    )
    Assert-SharedControl
    if (Test-NonEmptyDirectory (Get-ArtifactPath 'control')) {
        throw 'V20r density Control artifact is non-empty; shared Control must not be retrained.'
    }
    if (-not (Test-NonEmptyDirectory (Get-ArtifactPath 'treatment'))) {
        Invoke-RadarPython @(
            '-m', 'build.liquid_dynamics_v20',
            '--project-root', $projectPath,
            '--variant', 'density'
        )
        Invoke-RadarPython @(
            '-m', 'report.liquid_dynamics_v20_readiness',
            '--project-root', $projectPath,
            '--variant', 'density'
        )
    }
    else {
        Write-Host '[V20r] Treatment artifact is non-empty; immutable prepare generation is skipped.' -ForegroundColor Yellow
    }
    Assert-Prepared
    Invoke-RadarPython @(
        '-m', 'smoke.static_isolation_check',
        '--root', $srcRoot
    )
    Invoke-RadarPython @(
        '-m', 'smoke.liquid_dynamics_v20_contract_test',
        '--project-root', $projectPath,
        '--variant', 'density',
        '--prepared'
    )
}

function Invoke-Train {
    Write-Host '[V20r] stage=train; Treatment only; seed=42' -ForegroundColor Green
    Assert-Prepared
    $configPath = Get-ConfigPath 'treatment'
    $artifactPath = Get-ArtifactPath 'treatment'
    $summaryPath = Join-Path $artifactPath 'training-summary.json'
    $lastCheckpoint = Join-Path $artifactPath 'last_model.pt'
    if (Test-Path -LiteralPath $summaryPath -PathType Leaf) {
        $existing = Read-Utf8Json $summaryPath
        if ($existing.status -eq 'completed') {
            Assert-CompletedTreatment
            Write-Host '[V20r] Treatment already completed; preserving the formal artifact.' -ForegroundColor Yellow
            return
        }
    }
    $arguments = @(
        '-m', 'train.tasks',
        '--project-root', $projectPath,
        '--config', $configPath
    )
    if (Test-NonEmptyDirectory $artifactPath) {
        Assert-File $lastCheckpoint 'V20r Treatment resumable last checkpoint'
        $arguments += @('--resume-checkpoint', $lastCheckpoint)
        Write-Host ('[V20r] Treatment resumes from ' + $lastCheckpoint) -ForegroundColor Yellow
    }
    Invoke-RadarPython $arguments
    Assert-CompletedTreatment
}

function Invoke-Evaluate {
    Write-Host '[V20r] stage=evaluate; shared V20 Control versus density Treatment' -ForegroundColor Green
    Assert-Prepared
    Assert-CompletedTreatment
    Invoke-RadarPython @(
        '-m', 'report.liquid_dynamics_v20_paired',
        '--project-root', $projectPath,
        '--variant', 'density'
    )
    Assert-File $pairedReportPath 'V20r density paired validation report'
    $report = Read-Utf8Json $pairedReportPath
    if (
        $report.test -ne $notExecuted -or
        $report.OOF -ne $notExecuted -or
        $report.full_cascade -ne $notExecuted -or
        $report.formal_rejection -ne $notExecuted -or
        $report.stable -ne $notExecuted -or
        $report.P95 -ne $notExecuted -or
        $report.deployment_eligible -ne $false -or
        $report.shared_control_reuse.shared_control_retrained -ne $false
    ) {
        throw ('V20r report violates shared-Control or sealed-test scope: ' + $pairedReportPath)
    }
}

try {
    Write-Host ('[V20r] project=' + $projectPath) -ForegroundColor Green
    Write-Host ('[V20r] requested stage=' + $Stage) -ForegroundColor Green
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
    Write-Host '[V20r] requested stages completed. Only density Treatment is trained; OOF, test, cascade, rejection, stable, P95 and deployment remain unexecuted.' -ForegroundColor Green
}
catch {
    [Console]::Error.WriteLine('[V20r] density-weighted pipeline stopped: ' + $_.Exception.Message)
    exit 1
}
