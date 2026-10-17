[CmdletBinding()]
param(
    [ValidateSet('prepare', 'treatment', 'evaluate', 'all')]
    [string]$Stage = 'prepare',
    [string]$ProjectRoot = ''
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$utf8NoBom = [System.Text.UTF8Encoding]::new($false)
[Console]::InputEncoding = $utf8NoBom
[Console]::OutputEncoding = $utf8NoBom
$OutputEncoding = $utf8NoBom

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$srcRoot = Split-Path -Parent $scriptRoot
$defaultProjectRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $srcRoot))
$projectPath = [System.IO.Path]::GetFullPath(
    $(if ([string]::IsNullOrWhiteSpace($ProjectRoot)) { $defaultProjectRoot } else { $ProjectRoot })
)
$python = Join-Path $projectPath '.venv\Scripts\python.exe'
$configPath = Join-Path $projectPath 'config\RADAR-LN3\RADAR-LN3-V22a-cq08-liquid-known-gate-rd-temporal-residual-seed43-config.json'
$readinessPath = Join-Path $projectPath 'src-v3\processed\report\RADAR-LN3-V22a-known-gate-rd-temporal-residual-training-readiness.json'
$artifactPath = Join-Path $projectPath 'artifacts\RADAR-LN3-V22a-cq08-liquid-known-gate-rd-temporal-residual-seed43-formal'
$summaryPath = Join-Path $artifactPath 'training-summary.json'
$checkpointPath = Join-Path $artifactPath 'best_model.pt'
$lastCheckpointPath = Join-Path $artifactPath 'last_model.pt'
$reportPath = Join-Path $projectPath 'src-v3\processed\report\RADAR-LN3-V22a-known-gate-rd-temporal-residual-paired-validation.json'
$notExecuted = -join ([char[]](0x672A, 0x6267, 0x884C))

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw ('Project Python interpreter is missing: ' + $python)
}

function Invoke-RadarPython {
    param([Parameter(Mandatory = $true)][string[]]$Arguments)
    $runtimeArguments = @('-X', 'utf8') + $Arguments
    $display = $python + ' ' + ($runtimeArguments -join ' ')
    Write-Host ('> ' + $display) -ForegroundColor Cyan
    Push-Location $srcRoot
    try {
        & $python @runtimeArguments | Out-Host
        $exitCode = [int]$LASTEXITCODE
    }
    finally {
        Pop-Location
    }
    if ($exitCode -ne 0) {
        throw ('Python command failed with exit code ' + $exitCode + ': ' + $display)
    }
}

function Assert-File {
    param([Parameter(Mandatory = $true)][string]$Path, [Parameter(Mandatory = $true)][string]$Description)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw ($Description + ' is missing: ' + $Path)
    }
}

function Read-Json {
    param([Parameter(Mandatory = $true)][string]$Path)
    return ConvertFrom-Json -InputObject ([System.IO.File]::ReadAllText($Path, [System.Text.UTF8Encoding]::new($false, $true)))
}

function Assert-Readiness {
    Assert-File $readinessPath 'V22a readiness'
    $readiness = Read-Json $readinessPath
    $failed = @($readiness.checks.PSObject.Properties | Where-Object { $_.Value -ne $true })
    if (
        $readiness.status -ne 'passed' -or
        $readiness.schema_version -ne 'RADAR-LN3-V22a-known-gate-rd-temporal-residual-training-readiness-v1' -or
        $readiness.validation_used_for_readiness_gate -ne $false -or
        [int]$readiness.test_payload_access_count -ne 0 -or
        $readiness.test -ne $notExecuted -or
        $readiness.deployment_eligible -ne $false -or
        $failed.Count -ne 0
    ) {
        throw ('V22a readiness is stale or failed: ' + $readinessPath)
    }
}

function Assert-CompletedTraining {
    Assert-File $summaryPath 'V22a training summary'
    Assert-File $checkpointPath 'V22a best checkpoint'
    $summary = Read-Json $summaryPath
    if (
        $summary.status -ne 'completed' -or
        $summary.variant -ne 'v22a_rd_temporal_residual' -or
        [int]$summary.best_epoch -lt 1 -or
        $summary.full_parameter_training -ne $true -or
        $summary.all_model_parameters_trainable -ne $true -or
        $summary.paired_initialization.frozen_v21e_initial_model_state_reproduced -ne $true -or
        $summary.paired_initialization.real_training_probe_residual_columns_receive_gradient -ne $true -or
        $summary.rd_temporal_residual_training_audit.feature_used_by_best_checkpoint -ne $true -or
        $summary.test_evaluation -ne 'not_executed' -or
        $summary.oof_threshold_calibration -ne 'not_executed' -or
        $summary.formal_rejection -ne 'not_executed' -or
        $summary.stable_evaluation -ne 'not_executed' -or
        $summary.p95_latency -ne 'not_executed' -or
        $summary.deployment_eligible -ne $false
    ) {
        throw ('V22a training summary violates the local contract: ' + $summaryPath)
    }
}

function Invoke-Prepare {
    Write-Host '[V22a] preparing RD current-event temporal residual experiment' -ForegroundColor Green
    Invoke-RadarPython @(
        '-m', 'py_compile',
        'radar_ln3\model.py',
        'radar_ln3\known_gate_features.py',
        'radar_ln3\known_gate.py',
        'build\known_gate_v22a.py',
        'train\known_gate_common.py',
        'train\known_gate_v22a.py',
        'report\known_gate_v22a.py'
    )
    Invoke-RadarPython @('-m', 'build.known_gate_v22a', '--project-root', $projectPath)
    Assert-Readiness
}

function Invoke-Treatment {
    Assert-Readiness
    if (Test-Path -LiteralPath $summaryPath -PathType Leaf) {
        $existingSummary = Read-Json $summaryPath
        if ($existingSummary.status -eq 'completed') {
            Assert-CompletedTraining
            Write-Host '[V22a] Treatment already completed.' -ForegroundColor Yellow
            return
        }
    }
    $arguments = @('-m', 'train.known_gate_v22a', '--project-root', $projectPath)
    if ((Test-Path -LiteralPath $artifactPath -PathType Container) -and
        $null -ne (Get-ChildItem -LiteralPath $artifactPath -Force | Select-Object -First 1)) {
        Assert-File $lastCheckpointPath 'V22a epoch-boundary resume checkpoint'
        $arguments += @('--resume-checkpoint', $lastCheckpointPath)
        Write-Host ('[V22a] resuming from ' + $lastCheckpointPath) -ForegroundColor Yellow
    }
    Invoke-RadarPython $arguments
    Assert-CompletedTraining
}

function Invoke-Evaluate {
    Assert-Readiness
    Assert-CompletedTraining
    Invoke-RadarPython @('-m', 'report.known_gate_v22a', '--project-root', $projectPath)
    Assert-File $reportPath 'V22a paired validation report'
    $report = Read-Json $reportPath
    if ($report.schema_version -ne 'RADAR-LN3-V22a-known-gate-rd-temporal-residual-paired-validation-v1' -or
        $report.paired_training_contract.epoch_parent_order_hashes_exactly_equal -ne $true -or
        $report.rd_temporal_residual_validation_activity.single_or_missing_residual_is_exact_zero -ne $true -or
        $report.deployment_eligible -ne $false) {
        throw ('V22a paired validation report violates the local contract: ' + $reportPath)
    }
}

try {
    Write-Host ('[V22a] project=' + $projectPath) -ForegroundColor Green
    Write-Host ('[V22a] stage=' + $Stage) -ForegroundColor Green
    switch ($Stage) {
        'prepare' { Invoke-Prepare }
        'treatment' { Invoke-Treatment }
        'evaluate' { Invoke-Evaluate }
        'all' { Invoke-Prepare; Invoke-Treatment; Invoke-Evaluate }
    }
    Write-Host '[V22a] completed requested stages. Unknown, OOF, test, rejection, stable and P95 remain unexecuted.' -ForegroundColor Green
}
catch {
    [Console]::Error.WriteLine('[V22a] pipeline stopped: ' + $_.Exception.Message)
    exit 1
}
