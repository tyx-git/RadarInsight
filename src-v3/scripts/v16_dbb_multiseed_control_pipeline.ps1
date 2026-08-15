[CmdletBinding()]
param(
    [ValidateSet('prepare', 'train', 'verify', 'all')]
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
$readinessSchema = 'RADAR-LN3-V20-density-weighted-multiseed-training-readiness-v1'
$pairedProtocol = 'v20r-dbb-bounded-observation-control-versus-density-weighted-treatment-v1'
$newSeeds = @(43, 44)
$notExecuted = -join ([char[]](0x672A, 0x6267, 0x884C))
$seed42ControlArtifact = Join-Path $projectPath (
    'artifacts\RADAR-LN3-V20-cq08-liquid-dbb-specialist-zero-delta-skip-' +
    'control-seed42-formal'
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
    param([Parameter(Mandatory = $true)][int]$Seed)
    return Join-Path $configRoot (
        'RADAR-LN3-V20r-cq08-liquid-dbb-specialist-density-weighted-' +
        'control-seed' + $Seed + '-config.json'
    )
}

function Get-ArtifactPath {
    param([Parameter(Mandatory = $true)][int]$Seed)
    return Join-Path $projectPath (
        'artifacts\RADAR-LN3-V20r-cq08-liquid-dbb-specialist-density-weighted-' +
        'control-seed' + $Seed + '-formal'
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
    if ((@($readiness.seed_set) -join ',') -ne '42,43,44') {
        throw 'V20r multiseed seed set must be exactly 42,43,44.'
    }
    foreach ($seed in $newSeeds) {
        $key = 'seed' + $seed + '_control'
        $property = $readiness.paired_configs.PSObject.Properties[$key]
        if ($null -eq $property) {
            throw ('Readiness is missing config binding ' + $key)
        }
        $binding = $property.Value
        $configPath = Get-ConfigPath $seed
        Assert-File $configPath ('V16-core Control seed' + $seed + ' config')
        if (
            [System.IO.Path]::GetFullPath([string]$binding.path) -ne
                [System.IO.Path]::GetFullPath($configPath) -or
            [string]$binding.sha256 -ne (Get-FileHashLower $configPath)
        ) {
            throw ('Readiness config binding differs for Control seed' + $seed)
        }
    }
    return $readiness
}

function Assert-CompletedSeed42Control {
    $summaryPath = Join-Path $seed42ControlArtifact 'training-summary.json'
    Assert-File $summaryPath 'Reused seed42 V16-core Control summary'
    $summary = Read-Utf8Json $summaryPath
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
        $summary.deployment_eligible -ne $false
    ) {
        throw 'Reused seed42 V16-core Control is incomplete or unsealed.'
    }
}

function Assert-CompletedControl {
    param([Parameter(Mandatory = $true)][int]$Seed)

    $artifactPath = Get-ArtifactPath $Seed
    $summaryPath = Join-Path $artifactPath 'training-summary.json'
    Assert-File $summaryPath ('V16-core Control seed' + $Seed + ' summary')
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
        $summary.paired_initialization.paired_arm -ne 'control' -or
        $summary.paired_initialization.shared_parameters_and_buffers_exactly_equal -ne $true
    ) {
        throw ('V16-core Control seed' + $Seed + ' is incomplete or violates pairing.')
    }
    Assert-File ([string]$binding.path) ('Control seed' + $Seed + ' readiness binding')
    if ([string]$binding.sha256 -ne (Get-FileHashLower ([string]$binding.path))) {
        throw ('Control seed' + $Seed + ' readiness hash differs.')
    }
    Assert-File (Join-Path $artifactPath 'best_model.pt') ('Control seed' + $Seed + ' best checkpoint')
    Assert-File (Join-Path $artifactPath 'last_model.pt') ('Control seed' + $Seed + ' last checkpoint')
    for ($epoch = 1; $epoch -le 20; $epoch++) {
        Assert-File (
            Join-Path $artifactPath ('epochs\epoch-' + $epoch.ToString('000') + '.pt')
        ) ('Control seed' + $Seed + ' epoch checkpoint')
    }
}

function Invoke-Prepare {
    Write-Host '[V16 multiseed] prepare seed43/44 V16-core Controls' -ForegroundColor Green
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
    Assert-CompletedSeed42Control
    Invoke-RadarPython @(
        '-m', 'smoke.static_isolation_check',
        '--root', $srcRoot
    )
    Invoke-RadarPython @(
        '-m', 'smoke.liquid_dynamics_v20_multiseed_contract_test',
        '--project-root', $projectPath
    )
}

function Invoke-TrainControl {
    param([Parameter(Mandatory = $true)][int]$Seed)

    $configPath = Get-ConfigPath $Seed
    $artifactPath = Get-ArtifactPath $Seed
    $summaryPath = Join-Path $artifactPath 'training-summary.json'
    $lastCheckpoint = Join-Path $artifactPath 'last_model.pt'
    if (Test-Path -LiteralPath $summaryPath -PathType Leaf) {
        $existing = Read-Utf8Json $summaryPath
        if ($existing.status -eq 'completed') {
            Assert-CompletedControl $Seed
            Write-Host ('[V16 multiseed] Control seed' + $Seed + ' already completed.') -ForegroundColor Yellow
            return
        }
    }
    $arguments = @(
        '-m', 'train.tasks',
        '--project-root', $projectPath,
        '--config', $configPath
    )
    if (Test-NonEmptyDirectory $artifactPath) {
        Assert-File $lastCheckpoint ('Control seed' + $Seed + ' resumable checkpoint')
        $arguments += @('--resume-checkpoint', $lastCheckpoint)
        Write-Host ('[V16 multiseed] resuming Control seed' + $Seed) -ForegroundColor Yellow
    }
    Invoke-RadarPython $arguments
    Assert-CompletedControl $Seed
}

function Invoke-Train {
    $null = Assert-Readiness
    Assert-CompletedSeed42Control
    foreach ($seed in $newSeeds) {
        Write-Host ('[V16 multiseed] training bounded Control seed' + $seed) -ForegroundColor Green
        Invoke-TrainControl $seed
    }
}

function Invoke-Verify {
    $null = Assert-Readiness
    Assert-CompletedSeed42Control
    foreach ($seed in $newSeeds) {
        Assert-CompletedControl $seed
    }
    Write-Host '[V16 multiseed] Controls 42/43/44 are complete; density Treatments remain separate.' -ForegroundColor Green
}

try {
    Write-Host ('[V16 multiseed] project=' + $projectPath) -ForegroundColor Green
    Write-Host ('[V16 multiseed] stage=' + $Stage + '; new seeds=43,44; seed42=reused') -ForegroundColor Green
    switch ($Stage) {
        'prepare' { Invoke-Prepare }
        'train' { Invoke-Train }
        'verify' { Invoke-Verify }
        'all' {
            Invoke-Prepare
            Invoke-Train
            Invoke-Verify
        }
    }
    Write-Host '[V16 multiseed] requested stages completed; test/OOF/cascade/rejection/stable/P95 remain unexecuted.' -ForegroundColor Green
}
catch {
    [Console]::Error.WriteLine('[V16 multiseed] pipeline stopped: ' + $_.Exception.Message)
    exit 1
}
