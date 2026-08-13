[CmdletBinding()]
param(
    [ValidateSet('prepare', 'train', 'evaluate', 'all')]
    [string]$Stage = 'all'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$srcRoot = Split-Path -Parent $scriptRoot
$projectRoot = Split-Path -Parent $srcRoot
$python = Join-Path $projectRoot '.venv\Scripts\python.exe'
$configRoot = Join-Path $projectRoot 'config\RADAR-LN3'
$readinessPath = Join-Path $projectRoot (
    'src-v3\processed\report\RADAR-LN3-V18-broadband-multiscale-' +
    'micro-doppler21-pruned-train-readiness.json'
)
$pairedReportPath = Join-Path $projectRoot (
    'src-v3\processed\report\RADAR-LN3-V18-broadband-multiscale-' +
    'micro-doppler21-pruned-paired-validation.json'
)
$notExecuted = -join ([char[]](0x672A, 0x6267, 0x884C))
$readinessSchema = (
    'RADAR-LN3-V18-broadband-multiscale-micro-doppler-' +
    'training-readiness-v2-pruned21'
)

function Invoke-RadarPython([string[]]$Arguments) {
    $displayCommand = $python + ' ' + ($Arguments -join ' ')
    Write-Host ('> ' + $displayCommand) -ForegroundColor Cyan
    Push-Location $srcRoot
    try {
        & $python @Arguments
    }
    finally {
        Pop-Location
    }
    if ($LASTEXITCODE -ne 0) {
        throw (
            'Python command failed with exit code ' + $LASTEXITCODE +
            ': ' + $displayCommand
        )
    }
}

function Assert-File([string]$Path, [string]$Description) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw ($Description + ' is missing: ' + $Path)
    }
}

function Get-ConfigPath([string]$Arm) {
    return Join-Path $configRoot (
        'RADAR-LN3-V18-cq08-liquid-dbb-specialist-rd-physical4-' +
        'broadband-multiscale-micro-doppler21-pruned-' + $Arm + '-seed42-config.json'
    )
}

function Get-ArtifactRoot([string]$Arm) {
    return Join-Path $projectRoot (
        'artifacts\RADAR-LN3-V18-cq08-liquid-dbb-specialist-rd-physical4-' +
        'broadband-multiscale-micro-doppler21-pruned-' + $Arm + '-seed42-formal'
    )
}

function Assert-Readiness {
    Assert-File $readinessPath 'V18 train-only broadband micro-Doppler readiness report'
    $readiness = Get-Content -LiteralPath $readinessPath -Raw -Encoding UTF8 | ConvertFrom-Json
    if (-not ($readiness.PSObject.Properties.Name -contains 'status')) {
        throw ('V18 readiness report has no status field: ' + $readinessPath)
    }
    if (
        $readiness.status -ne 'passed' -or
        $readiness.schema_version -ne $readinessSchema -or
        $readiness.validation_used_for_readiness_gate -ne $false -or
        $readiness.test -ne $notExecuted
    ) {
        throw ('V18 readiness is not passed/train-only/test-sealed: ' + $readinessPath)
    }
    $failedChecks = @(
        $readiness.checks.PSObject.Properties |
            Where-Object { $_.Value -ne $true }
    )
    if ($failedChecks.Count -gt 0) {
        throw ('V18 readiness contains a failed check: ' + $readinessPath)
    }
    foreach ($arm in @('control', 'treatment')) {
        $configPath = Get-ConfigPath $arm
        Assert-File $configPath ('V18 ' + $arm + ' config')
        $binding = $readiness.paired_configs.$arm
        if (
            $null -eq $binding -or
            $binding.path -ne $configPath -or
            $binding.sha256 -ne (Get-FileHash -LiteralPath $configPath -Algorithm SHA256).Hash.ToLowerInvariant()
        ) {
            throw ('V18 readiness does not bind the ' + $arm + ' config: ' + $readinessPath)
        }
    }
}

function Assert-CompletedArm([string]$Arm) {
    $configPath = Get-ConfigPath $Arm
    $artifactRoot = Get-ArtifactRoot $Arm
    $summaryPath = Join-Path $artifactRoot 'training-summary.json'
    Assert-File $configPath ('V18 ' + $Arm + ' config')
    Assert-File $summaryPath ('V18 ' + $Arm + ' training summary')
    $summary = Get-Content -LiteralPath $summaryPath -Raw -Encoding UTF8 | ConvertFrom-Json
    foreach ($fieldName in @(
        'status', 'performance_status', 'config_name', 'task_name', 'stop_reason',
        'v18_training_readiness', 'test_evaluation', 'full_cascade_evaluation',
        'oof_rejection_calibration', 'stable_evaluation', 'p95_latency',
        'deployment_eligible'
    )) {
        if (-not ($summary.PSObject.Properties.Name -contains $fieldName)) {
            throw (
                'V18 ' + $Arm + ' summary is missing field ' + $fieldName +
                ': ' + $summaryPath
            )
        }
    }
    if (
        $summary.status -ne 'completed' -or
        $summary.task_name -ne 'dbb_specialist' -or
        $summary.stop_reason -ne 'epoch_limit'
    ) {
        throw (
            'V18 ' + $Arm + ' formal run is not a completed fixed-epoch DBB run: ' +
            $summaryPath
        )
    }
    if (
        $summary.v18_training_readiness.status -ne 'passed' -or
        $summary.v18_training_readiness.schema_version -ne $readinessSchema -or
        $summary.v18_training_readiness.validation_used_for_readiness_gate -ne $false -or
        $summary.v18_training_readiness.test -ne $notExecuted
    ) {
        throw (
            'V18 ' + $Arm + ' summary lacks a valid readiness binding: ' +
            $summaryPath
        )
    }
    if (
        $summary.test_evaluation -ne 'not_executed' -or
        $summary.full_cascade_evaluation -ne 'not_executed' -or
        $summary.oof_rejection_calibration -ne 'not_executed' -or
        $summary.stable_evaluation -ne 'not_executed' -or
        $summary.p95_latency -ne 'not_executed' -or
        $summary.deployment_eligible -ne $false
    ) {
        throw (
            'V18 ' + $Arm + ' summary violates local-only sealed-test discipline: ' +
            $summaryPath
        )
    }
    Assert-File (Join-Path $artifactRoot 'best_model.pt') ('V18 ' + $Arm + ' best checkpoint')
    Assert-File (Join-Path $artifactRoot 'last_model.pt') ('V18 ' + $Arm + ' last checkpoint')
    for ($epoch = 1; $epoch -le 20; $epoch++) {
        $epochName = 'epoch-' + $epoch.ToString('000') + '.pt'
        Assert-File (Join-Path $artifactRoot ('epochs\' + $epochName)) (
            'V18 ' + $Arm + ' epoch checkpoint'
        )
    }
}

function Invoke-Prepare {
    Invoke-RadarPython @(
        '-m', 'build.micro_doppler_v18',
        '--project-root', $projectRoot
    )
    Invoke-RadarPython @(
        '-m', 'smoke.micro_doppler_v18_contract_test',
        '--project-root', $projectRoot
    )
    try {
        Invoke-RadarPython @(
            '-m', 'report.micro_doppler_v18_readiness',
            '--project-root', $projectRoot
        )
    }
    catch {
        if (Test-Path -LiteralPath $readinessPath -PathType Leaf) {
            $blocked = Get-Content -LiteralPath $readinessPath -Raw -Encoding UTF8 | ConvertFrom-Json
            if ($blocked.status -eq 'blocked') {
                $failedNames = @(
                    $blocked.checks.PSObject.Properties |
                        Where-Object { $_.Value -ne $true } |
                        ForEach-Object { $_.Name }
                )
                throw (
                    'V18 train-only readiness blocked; failed checks=' +
                    ($failedNames -join ',') + '; report=' + $readinessPath
                )
            }
        }
        throw
    }
    Invoke-RadarPython @(
        '-m', 'smoke.static_isolation_check',
        '--root', $srcRoot
    )
    Assert-Readiness
}

function Invoke-TrainArm([string]$Arm) {
    $configPath = Get-ConfigPath $Arm
    $artifactRoot = Get-ArtifactRoot $Arm
    $summaryPath = Join-Path $artifactRoot 'training-summary.json'
    Assert-File $configPath ('V18 ' + $Arm + ' config')
    if (Test-Path -LiteralPath $summaryPath -PathType Leaf) {
        $existing = Get-Content -LiteralPath $summaryPath -Raw -Encoding UTF8 | ConvertFrom-Json
        if ($existing.status -eq 'completed') {
            Assert-CompletedArm $Arm
            Write-Host (
                '[V18] ' + $Arm +
                ' already completed; preserving the existing formal artifact.'
            ) -ForegroundColor Yellow
            return
        }
        if ($existing.status -ne 'failed') {
            throw (
                'V18 ' + $Arm + ' has an unsupported summary status; inspect: ' +
                $summaryPath
            )
        }
    }
    $arguments = @(
        '-m', 'train.tasks',
        '--project-root', $projectRoot,
        '--config', $configPath
    )
    $lastCheckpoint = Join-Path $artifactRoot 'last_model.pt'
    if (Test-Path -LiteralPath $artifactRoot) {
        $items = @(Get-ChildItem -LiteralPath $artifactRoot -Force)
        if ($items.Count -gt 0) {
            if (-not (Test-Path -LiteralPath $lastCheckpoint -PathType Leaf)) {
                throw (
                    'V18 ' + $Arm +
                    ' artifact is non-empty without a resumable last_model.pt; inspect: ' +
                    $artifactRoot
                )
            }
            $arguments += @('--resume-checkpoint', $lastCheckpoint)
            Write-Host (
                '[V18] ' + $Arm + ' resumes from ' + $lastCheckpoint
            ) -ForegroundColor Yellow
        }
    }
    Invoke-RadarPython $arguments
    Assert-CompletedArm $Arm
}

function Invoke-Train {
    Assert-Readiness
    Invoke-TrainArm 'control'
    Invoke-TrainArm 'treatment'
}

function Invoke-Evaluate {
    Assert-Readiness
    Assert-CompletedArm 'control'
    Assert-CompletedArm 'treatment'
    Invoke-RadarPython @(
        '-m', 'report.micro_doppler_v18_paired',
        '--project-root', $projectRoot
    )
    Assert-File $pairedReportPath 'V18 paired validation report'
    $report = Get-Content -LiteralPath $pairedReportPath -Raw -Encoding UTF8 | ConvertFrom-Json
    if (
        $report.test -ne $notExecuted -or
        $report.full_cascade -ne $notExecuted -or
        $report.deployment_eligible -ne $false
    ) {
        throw (
            'V18 paired report violates local-only sealed-test discipline: ' +
            $pairedReportPath
        )
    }
}

try {
    Assert-File $python 'Project Python interpreter'
    Write-Host ('[V18] project=' + $projectRoot) -ForegroundColor Green
    Write-Host ('[V18] stage=' + $Stage + '; task=DBB specialist; seed=42') -ForegroundColor Green
    Write-Host (
        '[V18] Control=RD physical4; Treatment=Control+21 pruned broadband multiscale ' +
        'micro-Doppler summaries; evidence profiles and PRT metadata are audit-only.'
    ) -ForegroundColor Green
    if ($Stage -in @('prepare', 'all')) {
        Invoke-Prepare
    }
    if ($Stage -in @('train', 'all')) {
        Invoke-Train
    }
    if ($Stage -in @('evaluate', 'all')) {
        Invoke-Evaluate
    }
    Write-Host (
        '[V18] requested stages completed. Full cascade, OOF, test, formal ' +
        'rejection, stable, P95, and deployment remain unexecuted.'
    ) -ForegroundColor Green
}
catch {
    Write-Error ('V18 broadband micro-Doppler paired pipeline stopped: ' + $_.Exception.Message)
    exit 1
}
