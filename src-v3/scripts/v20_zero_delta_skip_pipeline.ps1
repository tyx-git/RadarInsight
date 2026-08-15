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
    'src-v3\processed\report\RADAR-LN3-V20-zero-delta-skip-training-readiness.json'
)
$pairedReportPath = Join-Path $projectPath (
    'src-v3\processed\report\RADAR-LN3-V20-zero-delta-skip-paired-validation.json'
)
$notExecuted = -join ([char[]](0x672A, 0x6267, 0x884C))
$readinessSchema = 'RADAR-LN3-V20-zero-delta-skip-training-readiness-v1'
$pairedProtocol = 'v20-dbb-bounded-observation-control-versus-zero-delta-skip-treatment-v1'

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

function Get-ConfigPath {
    param([Parameter(Mandatory = $true)][ValidateSet('control', 'treatment')][string]$Arm)

    return Join-Path $configRoot (
        'RADAR-LN3-V20-cq08-liquid-dbb-specialist-zero-delta-skip-' +
        $Arm + '-seed42-config.json'
    )
}

function Get-ArtifactPath {
    param([Parameter(Mandatory = $true)][ValidateSet('control', 'treatment')][string]$Arm)

    return Join-Path $projectPath (
        'artifacts\RADAR-LN3-V20-cq08-liquid-dbb-specialist-zero-delta-skip-' +
        $Arm + '-seed42-formal'
    )
}

function Get-SummaryPath {
    param([Parameter(Mandatory = $true)][ValidateSet('control', 'treatment')][string]$Arm)

    return Join-Path (Get-ArtifactPath $Arm) 'training-summary.json'
}

function Get-FileHashLower {
    param([Parameter(Mandatory = $true)][string]$Path)

    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Assert-Prepared {
    Assert-File $readinessPath 'V20 skip train-only readiness report'
    $readiness = Read-Utf8Json $readinessPath
    if (-not ($readiness.PSObject.Properties.Name -contains 'checks')) {
        throw ('V20 readiness has no checks object: ' + $readinessPath)
    }
    $failedChecks = @(
        $readiness.checks.PSObject.Properties |
            Where-Object { $_.Value -ne $true }
    )
    if ($failedChecks.Count -gt 0) {
        throw (
            'V20 readiness contains failed checks=' +
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
        $readiness.deployment_eligible -ne $false
    ) {
        throw ('V20 readiness is not passed/train-only/test-sealed: ' + $readinessPath)
    }
    foreach ($arm in @('control', 'treatment')) {
        $configPath = Get-ConfigPath $arm
        Assert-File $configPath ('V20 ' + $arm + ' config')
        if (-not ($readiness.paired_configs.PSObject.Properties.Name -contains $arm)) {
            throw ('V20 readiness does not contain a ' + $arm + ' config binding')
        }
        $binding = $readiness.paired_configs.$arm
        $boundPath = [System.IO.Path]::GetFullPath([string]$binding.path)
        $expectedPath = [System.IO.Path]::GetFullPath($configPath)
        if (
            $boundPath -ne $expectedPath -or
            [string]$binding.sha256 -ne (Get-FileHashLower $configPath)
        ) {
            throw ('V20 readiness does not bind the ' + $arm + ' config')
        }
    }
}

function Assert-CompletedArm {
    param([Parameter(Mandatory = $true)][ValidateSet('control', 'treatment')][string]$Arm)

    $artifactPath = Get-ArtifactPath $Arm
    $summaryPath = Get-SummaryPath $Arm
    Assert-File $summaryPath ('V20 ' + $Arm + ' training summary')
    $summary = Read-Utf8Json $summaryPath
    $required = @(
        'status', 'performance_status', 'config_name', 'task_name', 'stop_reason',
        'v20_training_readiness', 'test_evaluation', 'full_cascade_evaluation',
        'oof_rejection_calibration', 'stable_evaluation', 'p95_latency',
        'deployment_eligible', 'history', 'best_epoch'
    )
    foreach ($fieldName in $required) {
        if (-not ($summary.PSObject.Properties.Name -contains $fieldName)) {
            throw ('V20 ' + $Arm + ' summary is missing field ' + $fieldName)
        }
    }
    if (
        $summary.status -ne 'completed' -or
        $summary.task_name -ne 'dbb_specialist' -or
        $summary.stop_reason -ne 'epoch_limit' -or
        $summary.test_evaluation -ne 'not_executed' -or
        $summary.full_cascade_evaluation -ne 'not_executed' -or
        $summary.oof_rejection_calibration -ne 'not_executed' -or
        $summary.stable_evaluation -ne 'not_executed' -or
        $summary.p95_latency -ne 'not_executed' -or
        $summary.deployment_eligible -ne $false
    ) {
        throw ('V20 ' + $Arm + ' formal run violates the local-only sealed-test contract')
    }
    $readiness = $summary.v20_training_readiness
    if (
        $readiness.status -ne 'passed' -or
        $readiness.schema_version -ne $readinessSchema -or
        $readiness.paired_protocol -ne $pairedProtocol -or
        $readiness.validation_used_for_readiness_gate -ne $false -or
        $readiness.test -ne $notExecuted
    ) {
        throw ('V20 ' + $Arm + ' summary has an invalid readiness binding')
    }
    Assert-File ([string]$readiness.path) ('V20 ' + $Arm + ' readiness binding')
    if ([string]$readiness.sha256 -ne (Get-FileHashLower ([string]$readiness.path))) {
        throw ('V20 ' + $Arm + ' readiness hash binding is invalid')
    }
    Assert-File (Join-Path $artifactPath 'best_model.pt') ('V20 ' + $Arm + ' best checkpoint')
    Assert-File (Join-Path $artifactPath 'last_model.pt') ('V20 ' + $Arm + ' last checkpoint')
    for ($epoch = 1; $epoch -le 20; $epoch++) {
        Assert-File (
            Join-Path $artifactPath ('epochs\epoch-' + $epoch.ToString('000') + '.pt')
        ) ('V20 ' + $Arm + ' epoch checkpoint')
    }
}

function Invoke-Prepare {
    Write-Host '[V20] stage=prepare; variant=zero-delta-skip' -ForegroundColor Green
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

    $artifactExists = (Test-NonEmptyDirectory (Get-ArtifactPath 'control')) -or
        (Test-NonEmptyDirectory (Get-ArtifactPath 'treatment'))
    if (-not $artifactExists) {
        Invoke-RadarPython @(
            '-m', 'build.liquid_dynamics_v20',
            '--project-root', $projectPath,
            '--variant', 'skip'
        )
        Invoke-RadarPython @(
            '-m', 'report.liquid_dynamics_v20_readiness',
            '--project-root', $projectPath,
            '--variant', 'skip'
        )
    }
    else {
        Write-Host '[V20] formal artifact is non-empty; prepare generation is immutable and skipped.' -ForegroundColor Yellow
    }

    Assert-Prepared
    Invoke-RadarPython @(
        '-m', 'smoke.static_isolation_check',
        '--root', $srcRoot
    )
    Invoke-RadarPython @(
        '-m', 'smoke.liquid_dynamics_v20_contract_test',
        '--project-root', $projectPath,
        '--prepared'
    )
}

function Invoke-TrainArm {
    param([Parameter(Mandatory = $true)][ValidateSet('control', 'treatment')][string]$Arm)

    $configPath = Get-ConfigPath $Arm
    $artifactPath = Get-ArtifactPath $Arm
    $summaryPath = Get-SummaryPath $Arm
    $lastCheckpoint = Join-Path $artifactPath 'last_model.pt'
    Assert-File $configPath ('V20 ' + $Arm + ' config')

    if (Test-Path -LiteralPath $summaryPath -PathType Leaf) {
        $existing = Read-Utf8Json $summaryPath
        if ($existing.status -eq 'completed') {
            Assert-CompletedArm $Arm
            Write-Host ('[V20] ' + $Arm + ' already completed; preserving the formal artifact.') -ForegroundColor Yellow
            return
        }
    }

    $arguments = @(
        '-m', 'train.tasks',
        '--project-root', $projectPath,
        '--config', $configPath
    )
    if (Test-NonEmptyDirectory $artifactPath) {
        Assert-File $lastCheckpoint ('V20 ' + $Arm + ' resumable last checkpoint')
        $arguments += @('--resume-checkpoint', $lastCheckpoint)
        Write-Host ('[V20] ' + $Arm + ' resumes from ' + $lastCheckpoint) -ForegroundColor Yellow
    }
    Invoke-RadarPython $arguments
    Assert-CompletedArm $Arm
}

function Invoke-Train {
    Write-Host '[V20] stage=train; task=DBB specialist; seed=42' -ForegroundColor Green
    Assert-Prepared
    Invoke-TrainArm 'control'
    Invoke-TrainArm 'treatment'
}

function Invoke-Evaluate {
    Write-Host '[V20] stage=evaluate; development validation only' -ForegroundColor Green
    Assert-Prepared
    Assert-CompletedArm 'control'
    Assert-CompletedArm 'treatment'
    Invoke-RadarPython @(
        '-m', 'report.liquid_dynamics_v20_paired',
        '--project-root', $projectPath,
        '--variant', 'skip'
    )
    Assert-File $pairedReportPath 'V20 paired validation report'
    $report = Read-Utf8Json $pairedReportPath
    if (
        $report.test -ne $notExecuted -or
        $report.OOF -ne $notExecuted -or
        $report.full_cascade -ne $notExecuted -or
        $report.formal_rejection -ne $notExecuted -or
        $report.stable -ne $notExecuted -or
        $report.P95 -ne $notExecuted -or
        $report.deployment_eligible -ne $false
    ) {
        throw ('V20 paired report violates local-only sealed-test discipline: ' + $pairedReportPath)
    }
}

try {
    Write-Host ('[V20] project=' + $projectPath) -ForegroundColor Green
    Write-Host ('[V20] requested stage=' + $Stage) -ForegroundColor Green
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
    Write-Host '[V20] requested stages completed. V20r, CU/Gate, OOF, test, formal rejection, stable, P95 and deployment remain unexecuted.' -ForegroundColor Green
}
catch {
    [Console]::Error.WriteLine('[V20] zero-delta-skip pipeline stopped: ' + $_.Exception.Message)
    exit 1
}
