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
    'RADAR-LN3-V21e-known-gate-causal-course-consistency-' +
    'multiseed-training-readiness.json'
)
$reportPath = Join-Path $projectPath (
    'src-v3\processed\report\' +
    'RADAR-LN3-V21e-known-gate-causal-course-consistency-' +
    'multiseed-paired-validation.json'
)
$readinessSchema = (
    'RADAR-LN3-V21e-known-gate-causal-course-consistency-' +
    'multiseed-training-readiness-v1'
)
$reportSchema = (
    'RADAR-LN3-V21e-known-gate-causal-course-consistency-' +
    'multiseed-paired-validation-v1'
)
$seeds = @(42, 43, 44)
$newTreatmentSeeds = @(42, 44)
$notExecuted = -join ([char[]](0x672A, 0x6267, 0x884C))

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

function Get-SummarySeed {
    param([Parameter(Mandatory = $true)][object]$Summary)
    $topLevel = $Summary.PSObject.Properties['seed']
    if ($null -ne $topLevel -and $null -ne $topLevel.Value) {
        return [int]$topLevel.Value
    }
    $loader = $Summary.PSObject.Properties['dataloader_contract']
    if ($null -ne $loader -and $null -ne $loader.Value) {
        $nested = $loader.Value.PSObject.Properties['seed']
        if ($null -ne $nested -and $null -ne $nested.Value) {
            return [int]$nested.Value
        }
    }
    $paired = $Summary.PSObject.Properties['paired_initialization']
    if ($null -ne $paired -and $null -ne $paired.Value) {
        $nested = $paired.Value.PSObject.Properties['seed']
        if ($null -ne $nested -and $null -ne $nested.Value) {
            return [int]$nested.Value
        }
    }
    throw 'Training summary does not expose a seed in any supported location.'
}

function Get-BaselineConfigPath {
    param([Parameter(Mandatory = $true)][int]$Seed)
    return Join-Path $configRoot (
        'RADAR-LN3-V21-cq08-liquid-known-gate-baseline-seed' +
        $Seed + '-config.json'
    )
}

function Get-BaselineArtifactPath {
    param([Parameter(Mandatory = $true)][int]$Seed)
    return Join-Path $projectPath (
        'artifacts\RADAR-LN3-V21-cq08-liquid-known-gate-baseline-seed' +
        $Seed + '-formal'
    )
}

function Get-TreatmentConfigPath {
    param([Parameter(Mandatory = $true)][int]$Seed)
    return Join-Path $configRoot (
        'RADAR-LN3-V21e-cq08-liquid-known-gate-causal-course-consistency-' +
        'seed' + $Seed + '-config.json'
    )
}

function Get-TreatmentArtifactPath {
    param([Parameter(Mandatory = $true)][int]$Seed)
    return Join-Path $projectPath (
        'artifacts\RADAR-LN3-V21e-cq08-liquid-known-gate-' +
        'causal-course-consistency-seed' + $Seed + '-formal'
    )
}

function Assert-Readiness {
    Assert-File $readinessPath 'V21e multiseed readiness'
    $readiness = Read-Utf8Json $readinessPath
    $failedChecks = @(
        $readiness.checks.PSObject.Properties |
            Where-Object { $_.Value -ne $true }
    )
    if (
        $readiness.status -ne 'passed' -or
        $readiness.schema_version -ne $readinessSchema -or
        (@($readiness.seed_set | ForEach-Object { [int]$_ }) -join ',') -ne '42,43,44' -or
        (@($readiness.new_training_seeds | ForEach-Object { [int]$_ }) -join ',') -ne '42,44' -or
        $readiness.validation_used_for_readiness_gate -ne $false -or
        $readiness.formal_training_at_report_generation -ne $notExecuted -or
        $readiness.test -ne $notExecuted -or
        $readiness.test_payload_access_count -ne 0 -or
        $readiness.deployment_eligible -ne $false -or
        $failedChecks.Count -ne 0
    ) {
        throw ('V21e multiseed readiness is stale, failed, or unsealed: ' + $readinessPath)
    }
    foreach ($seed in $newTreatmentSeeds) {
        $property = $readiness.treatment_configs.PSObject.Properties[[string]$seed]
        if ($null -eq $property) {
            throw ('Readiness is missing Treatment seed' + $seed + ' binding')
        }
        $binding = $property.Value
        $configPath = Get-TreatmentConfigPath $seed
        Assert-File $configPath ('V21e Treatment seed' + $seed + ' config')
        if (
            [System.IO.Path]::GetFullPath([string]$binding.path) -ne
                [System.IO.Path]::GetFullPath($configPath) -or
            [string]$binding.sha256 -ne (Get-FileHashLower $configPath)
        ) {
            throw ('V21e Treatment seed' + $seed + ' config binding differs')
        }
    }
    return $readiness
}

function Assert-CompletedRun {
    param(
        [Parameter(Mandatory = $true)][ValidateSet('baseline', 'treatment')][string]$Arm,
        [Parameter(Mandatory = $true)][int]$Seed
    )
    $artifactPath = if ($Arm -eq 'baseline') {
        Get-BaselineArtifactPath $Seed
    }
    else {
        Get-TreatmentArtifactPath $Seed
    }
    $configPath = if ($Arm -eq 'baseline') {
        Get-BaselineConfigPath $Seed
    }
    else {
        Get-TreatmentConfigPath $Seed
    }
    $summaryPath = Join-Path $artifactPath 'training-summary.json'
    Assert-File $configPath ($Arm + ' seed' + $Seed + ' config')
    Assert-File $summaryPath ($Arm + ' seed' + $Seed + ' summary')
    $summary = Read-Utf8Json $summaryPath
    $summarySeed = Get-SummarySeed $summary
    $expectedVariant = if ($Arm -eq 'baseline') { 'baseline' } else { 'v21e_course_prefix' }
    if (
        $summary.status -ne 'completed' -or
        $summary.task_name -ne 'known_gate' -or
        $summary.variant -ne $expectedVariant -or
        $summarySeed -ne $Seed -or
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
        $summary.deployment_eligible -ne $false -or
        [System.IO.Path]::GetFullPath([string]$summary.config_path) -ne
            [System.IO.Path]::GetFullPath($configPath)
    ) {
        throw ($Arm + ' seed' + $Seed + ' is incomplete or outside the sealed contract')
    }
    if ($Arm -eq 'treatment') {
        if (
            $summary.paired_initialization.paired_baseline -ne $true -or
            $summary.paired_initialization.shared_parameters_and_buffers_exactly_equal -ne $true -or
            $summary.paired_initialization.new_fusion_columns_exactly_zero -ne $true -or
            $summary.paired_initialization.initial_logits_exactly_equal -ne $true -or
            [int]$summary.paired_initialization.new_fusion_columns -ne 1 -or
            [int]$summary.paired_initialization.seed -ne $Seed
        ) {
            throw ('Treatment seed' + $Seed + ' pairing evidence is invalid')
        }
    }
    Assert-File (Join-Path $artifactPath 'best_model.pt') ($Arm + ' seed' + $Seed + ' best checkpoint')
    Assert-File (Join-Path $artifactPath 'last_model.pt') ($Arm + ' seed' + $Seed + ' last checkpoint')
    Assert-File (Join-Path $artifactPath 'sample_detail_information.json') ($Arm + ' seed' + $Seed + ' sample detail')
    for ($epoch = 1; $epoch -le 20; $epoch++) {
        Assert-File (
            Join-Path $artifactPath ('epochs\epoch-' + $epoch.ToString('000') + '.pt')
        ) ($Arm + ' seed' + $Seed + ' epoch checkpoint')
    }
}

function Invoke-Prepare {
    Write-Host '[V21e multiseed] validating frozen sources and generating seed42/44 Treatment configs' -ForegroundColor Green
    Invoke-RadarPython @(
        '-m', 'py_compile',
        'radar_ln3\known_gate.py',
        'train\known_gate_v21.py',
        'build\known_gate_v21e_multiseed.py',
        'report\known_gate_v21e_multiseed.py',
        'smoke\known_gate_v21e_multiseed_contract_test.py'
    )
    Invoke-RadarPython @(
        '-m', 'build.known_gate_v21e_multiseed',
        '--project-root', $projectPath
    )
    $null = Assert-Readiness
    foreach ($seed in $seeds) {
        Assert-CompletedRun 'baseline' $seed
    }
    Assert-CompletedRun 'treatment' 43
    Invoke-RadarPython @(
        '-m', 'smoke.static_isolation_check',
        '--root', $srcRoot
    )
    Invoke-RadarPython @(
        '-m', 'smoke.known_gate_v21e_multiseed_contract_test',
        '--project-root', $projectPath
    )
}

function Invoke-TrainTreatment {
    param([Parameter(Mandatory = $true)][int]$Seed)

    $configPath = Get-TreatmentConfigPath $Seed
    $artifactPath = Get-TreatmentArtifactPath $Seed
    $summaryPath = Join-Path $artifactPath 'training-summary.json'
    $lastCheckpoint = Join-Path $artifactPath 'last_model.pt'
    if (Test-Path -LiteralPath $summaryPath -PathType Leaf) {
        $existing = Read-Utf8Json $summaryPath
        if ($existing.status -eq 'completed') {
            Assert-CompletedRun 'treatment' $Seed
            Write-Host ('[V21e multiseed] Treatment seed' + $Seed + ' already completed.') -ForegroundColor Yellow
            return
        }
    }
    $arguments = @(
        '-m', 'train.known_gate_v21',
        '--project-root', $projectPath,
        '--config', $configPath
    )
    if (Test-NonEmptyDirectory $artifactPath) {
        Assert-File $lastCheckpoint ('Treatment seed' + $Seed + ' resumable checkpoint')
        $arguments += @('--resume-checkpoint', $lastCheckpoint)
        Write-Host ('[V21e multiseed] resuming Treatment seed' + $Seed) -ForegroundColor Yellow
    }
    Invoke-RadarPython $arguments
    Assert-CompletedRun 'treatment' $Seed
}

function Invoke-Train {
    $null = Assert-Readiness
    foreach ($seed in $seeds) {
        Assert-CompletedRun 'baseline' $seed
    }
    Assert-CompletedRun 'treatment' 43
    foreach ($seed in $newTreatmentSeeds) {
        Write-Host ('[V21e multiseed] full 20-epoch Treatment seed' + $seed) -ForegroundColor Green
        Invoke-TrainTreatment $seed
    }
}

function Invoke-Evaluate {
    $null = Assert-Readiness
    foreach ($seed in $seeds) {
        Assert-CompletedRun 'baseline' $seed
        Assert-CompletedRun 'treatment' $seed
    }
    Invoke-RadarPython @(
        '-m', 'report.known_gate_v21e_multiseed',
        '--project-root', $projectPath
    )
    Assert-File $reportPath 'V21e multiseed paired validation report'
    $report = Read-Utf8Json $reportPath
    if (
        $report.schema_version -ne $reportSchema -or
        (@($report.seed_set | ForEach-Object { [int]$_ }) -join ',') -ne '42,43,44' -or
        $report.test -ne $notExecuted -or
        $report.OOF -ne $notExecuted -or
        $report.full_cascade -ne $notExecuted -or
        $report.formal_rejection -ne $notExecuted -or
        $report.stable -ne $notExecuted -or
        $report.P95 -ne $notExecuted -or
        $report.deployment_eligible -ne $false -or
        $report.readiness.sha256 -ne (Get-FileHashLower $readinessPath)
    ) {
        throw ('V21e multiseed report violates the frozen local/test-sealed contract: ' + $reportPath)
    }
}

try {
    Write-Host ('[V21e multiseed] project=' + $projectPath) -ForegroundColor Green
    Write-Host ('[V21e multiseed] stage=' + $Stage + '; seeds=42,43,44; controls=reused; Treatment43=reused; Treatment42/44=full20epoch') -ForegroundColor Green
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
    Write-Host '[V21e multiseed] requested stages completed; test/OOF/cascade/rejection/stable/P95 remain unexecuted.' -ForegroundColor Green
}
catch {
    [Console]::Error.WriteLine('[V21e multiseed] pipeline stopped: ' + $_.Exception.Message)
    exit 1
}
