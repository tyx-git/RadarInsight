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
    'src-v3\processed\report\' +
    'RADAR-LN3-V20-density-weighted-multiseed-training-readiness.json'
)
$reportPath = Join-Path $projectPath (
    'src-v3\processed\report\' +
    'RADAR-LN3-V20-density-weighted-multiseed-paired-validation.json'
)
$readinessSchema = 'RADAR-LN3-V20-density-weighted-multiseed-training-readiness-v1'
$reportSchema = 'RADAR-LN3-V20-density-weighted-multiseed-paired-validation-v1'
$pairedProtocol = 'v20r-dbb-bounded-observation-control-versus-density-weighted-treatment-v1'
$newSeeds = @(43, 44)
$notExecuted = -join ([char[]](0x672A, 0x6267, 0x884C))
$seed42TreatmentArtifact = Join-Path $projectPath (
    'artifacts\RADAR-LN3-V20r-cq08-liquid-dbb-specialist-density-weighted-' +
    'treatment-seed42-formal'
)

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
    param(
        [Parameter(Mandatory = $true)][ValidateSet('control', 'treatment')][string]$Arm,
        [Parameter(Mandatory = $true)][int]$Seed
    )
    return Join-Path $configRoot (
        'RADAR-LN3-V20r-cq08-liquid-dbb-specialist-density-weighted-' +
        $Arm + '-seed' + $Seed + '-config.json'
    )
}

function Get-ArtifactPath {
    param(
        [Parameter(Mandatory = $true)][ValidateSet('control', 'treatment')][string]$Arm,
        [Parameter(Mandatory = $true)][int]$Seed
    )
    return Join-Path $projectPath (
        'artifacts\RADAR-LN3-V20r-cq08-liquid-dbb-specialist-density-weighted-' +
        $Arm + '-seed' + $Seed + '-formal'
    )
}

function Assert-Readiness {
    Assert-File $readinessPath 'V20r multiseed train-only readiness'
    $readiness = Read-Utf8Json $readinessPath
    $failedChecks = @(
        $readiness.checks.PSObject.Properties |
            Where-Object { $_.Value -ne $true }
    )
    if (
        $readiness.status -ne 'passed' -or
        $readiness.schema_version -ne $readinessSchema -or
        $readiness.paired_protocol -ne $pairedProtocol -or
        $readiness.validation_used_for_readiness_gate -ne $false -or
        $readiness.formal_training_at_report_generation -ne $notExecuted -or
        $readiness.test -ne $notExecuted -or
        $readiness.test_payload_access_count -ne 0 -or
        $readiness.deployment_eligible -ne $false -or
        $failedChecks.Count -ne 0
    ) {
        throw ('V20r multiseed readiness is stale, failed, or unsealed: ' + $readinessPath)
    }
    foreach ($seed in $newSeeds) {
        foreach ($arm in @('control', 'treatment')) {
            $key = 'seed' + $seed + '_' + $arm
            $property = $readiness.paired_configs.PSObject.Properties[$key]
            if ($null -eq $property) {
                throw ('Readiness is missing config binding ' + $key)
            }
            $binding = $property.Value
            $configPath = Get-ConfigPath $arm $seed
            Assert-File $configPath ('V20r ' + $arm + ' seed' + $seed + ' config')
            if (
                [System.IO.Path]::GetFullPath([string]$binding.path) -ne
                    [System.IO.Path]::GetFullPath($configPath) -or
                [string]$binding.sha256 -ne (Get-FileHashLower $configPath)
            ) {
                throw ('Readiness config binding differs for ' + $arm + ' seed' + $seed)
            }
        }
    }
    return $readiness
}

function Assert-CompletedArm {
    param(
        [Parameter(Mandatory = $true)][ValidateSet('control', 'treatment')][string]$Arm,
        [Parameter(Mandatory = $true)][int]$Seed
    )

    $artifactPath = Get-ArtifactPath $Arm $Seed
    $summaryPath = Join-Path $artifactPath 'training-summary.json'
    Assert-File $summaryPath ('V20r ' + $Arm + ' seed' + $Seed + ' summary')
    $summary = Read-Utf8Json $summaryPath
    $binding = $summary.v20_training_readiness
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
        $binding.paired_protocol -ne $pairedProtocol -or
        $summary.paired_initialization.seed -ne $Seed -or
        $summary.paired_initialization.paired_arm -ne $Arm -or
        $summary.paired_initialization.shared_parameters_and_buffers_exactly_equal -ne $true
    ) {
        throw ('V20r ' + $Arm + ' seed' + $Seed + ' is incomplete or violates pairing.')
    }
    Assert-File ([string]$binding.path) ($Arm + ' seed' + $Seed + ' readiness binding')
    if ([string]$binding.sha256 -ne (Get-FileHashLower ([string]$binding.path))) {
        throw ($Arm + ' seed' + $Seed + ' readiness hash differs.')
    }
    Assert-File (Join-Path $artifactPath 'best_model.pt') ($Arm + ' seed' + $Seed + ' best checkpoint')
    Assert-File (Join-Path $artifactPath 'last_model.pt') ($Arm + ' seed' + $Seed + ' last checkpoint')
    for ($epoch = 1; $epoch -le 20; $epoch++) {
        Assert-File (
            Join-Path $artifactPath ('epochs\epoch-' + $epoch.ToString('000') + '.pt')
        ) ($Arm + ' seed' + $Seed + ' epoch checkpoint')
    }
}

function Assert-Seed42Treatment {
    $summaryPath = Join-Path $seed42TreatmentArtifact 'training-summary.json'
    Assert-File $summaryPath 'Reused seed42 density Treatment summary'
    $summary = Read-Utf8Json $summaryPath
    if (
        $summary.status -ne 'completed' -or
        $summary.task_name -ne 'dbb_specialist' -or
        @($summary.history).Count -ne 20 -or
        $summary.test_evaluation -ne 'not_executed' -or
        $summary.deployment_eligible -ne $false
    ) {
        throw 'Reused seed42 density Treatment is incomplete or unsealed.'
    }
}

function Assert-AllControls {
    foreach ($seed in $newSeeds) {
        Assert-CompletedArm 'control' $seed
    }
}

function Invoke-Prepare {
    Write-Host '[V20r multiseed] prepare confirmation for seeds 43/44' -ForegroundColor Green
    Invoke-RadarPython @(
        '-m', 'py_compile',
        'radar_ln3\task_models.py',
        'train\tasks.py',
        'build\liquid_dynamics_v20.py',
        'build\liquid_dynamics_v20_multiseed.py',
        'report\liquid_dynamics_v20_paired.py',
        'report\liquid_dynamics_v20_multiseed.py',
        'smoke\liquid_dynamics_v20_multiseed_contract_test.py'
    )
    Invoke-RadarPython @(
        '-m', 'build.liquid_dynamics_v20_multiseed',
        '--project-root', $projectPath
    )
    $null = Assert-Readiness
    Assert-Seed42Treatment
    Assert-AllControls
    Invoke-RadarPython @(
        '-m', 'smoke.static_isolation_check',
        '--root', $srcRoot
    )
    Invoke-RadarPython @(
        '-m', 'smoke.liquid_dynamics_v20_multiseed_contract_test',
        '--project-root', $projectPath
    )
}

function Invoke-TrainTreatment {
    param([Parameter(Mandatory = $true)][int]$Seed)

    $configPath = Get-ConfigPath 'treatment' $Seed
    $artifactPath = Get-ArtifactPath 'treatment' $Seed
    $summaryPath = Join-Path $artifactPath 'training-summary.json'
    $lastCheckpoint = Join-Path $artifactPath 'last_model.pt'
    if (Test-Path -LiteralPath $summaryPath -PathType Leaf) {
        $existing = Read-Utf8Json $summaryPath
        if ($existing.status -eq 'completed') {
            Assert-CompletedArm 'treatment' $Seed
            Write-Host ('[V20r multiseed] Treatment seed' + $Seed + ' already completed.') -ForegroundColor Yellow
            return
        }
    }
    $arguments = @(
        '-m', 'train.tasks',
        '--project-root', $projectPath,
        '--config', $configPath
    )
    if (Test-NonEmptyDirectory $artifactPath) {
        Assert-File $lastCheckpoint ('Treatment seed' + $Seed + ' resumable checkpoint')
        $arguments += @('--resume-checkpoint', $lastCheckpoint)
        Write-Host ('[V20r multiseed] resuming Treatment seed' + $Seed) -ForegroundColor Yellow
    }
    Invoke-RadarPython $arguments
    Assert-CompletedArm 'treatment' $Seed
}

function Invoke-Train {
    $null = Assert-Readiness
    Assert-Seed42Treatment
    Assert-AllControls
    foreach ($seed in $newSeeds) {
        Write-Host ('[V20r multiseed] training density Treatment seed' + $seed) -ForegroundColor Green
        Invoke-TrainTreatment $seed
    }
}

function Invoke-Evaluate {
    $readiness = Assert-Readiness
    Assert-Seed42Treatment
    Assert-AllControls
    foreach ($seed in $newSeeds) {
        Assert-CompletedArm 'treatment' $seed
    }
    Invoke-RadarPython @(
        '-m', 'report.liquid_dynamics_v20_multiseed',
        '--project-root', $projectPath
    )
    Assert-File $reportPath 'V20r multiseed paired validation report'
    $report = Read-Utf8Json $reportPath
    if (
        $report.schema_version -ne $reportSchema -or
        $report.paired_protocol -ne $pairedProtocol -or
        (@($report.seed_set) -join ',') -ne '42,43,44' -or
        $report.test -ne $notExecuted -or
        $report.OOF -ne $notExecuted -or
        $report.full_cascade -ne $notExecuted -or
        $report.formal_rejection -ne $notExecuted -or
        $report.stable -ne $notExecuted -or
        $report.P95 -ne $notExecuted -or
        $report.deployment_eligible -ne $false -or
        $report.readiness.sha256 -ne (Get-FileHashLower $readinessPath)
    ) {
        throw ('V20r multiseed report violates the frozen local/test-sealed contract: ' + $reportPath)
    }
}

try {
    Write-Host ('[V20r multiseed] project=' + $projectPath) -ForegroundColor Green
    Write-Host ('[V20r multiseed] stage=' + $Stage + '; seeds=42,43,44; seed42=reused') -ForegroundColor Green
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
    Write-Host '[V20r multiseed] requested stages completed; test/OOF/cascade/rejection/stable/P95 remain unexecuted.' -ForegroundColor Green
}
catch {
    [Console]::Error.WriteLine('[V20r multiseed] pipeline stopped: ' + $_.Exception.Message)
    exit 1
}
