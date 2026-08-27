[CmdletBinding()]
param(
    [ValidateSet('prepare', 'smoke', 'l1', 'l2', 'train', 'calibrate', 'all')]
    [string]$Stage = 'all',
    [string]$ProjectRoot = ''
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$utf8NoBom = [System.Text.UTF8Encoding]::new($false)
[Console]::InputEncoding = $utf8NoBom
[Console]::OutputEncoding = $utf8NoBom
$OutputEncoding = $utf8NoBom

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$srcRoot = (Resolve-Path -LiteralPath (Split-Path -Parent $scriptRoot)).Path
$defaultProjectRoot = (Resolve-Path -LiteralPath (Split-Path -Parent $srcRoot)).Path
$projectPath = [System.IO.Path]::GetFullPath($(if ([string]::IsNullOrWhiteSpace($ProjectRoot)) { $defaultProjectRoot } else { $ProjectRoot }))
$python = Join-Path $projectPath '.venv\Scripts\python.exe'
$contractPath = Join-Path $projectPath 'data\processed\RADAR-LN3-V26c-cq08-l1-l2-known-only-oof-calibration-contract.json'
$readinessPath = Join-Path $projectPath 'src-v3\processed\report\RADAR-LN3-V26c-l1-l2-known-only-oof-calibration-readiness.json'
$artifactRoot = Join-Path $projectPath 'artifacts\RADAR-LN3-V26c-cq08-l1-l2-known-only-oof-calibration'
$reportPath = Join-Path $projectPath 'src-v3\processed\report\RADAR-LN3-V26c-l1-l2-known-only-cross-fitted-l3-calibration.json'
$detailPath = Join-Path $projectPath 'src-v3\processed\report\RADAR-LN3-V26c-l1-l2-known-only-cross-fitted-l3-parent-details.csv'
$minimumFreeGpuMemoryMiB = 4096
$maxReportBytes = 6 * 1024 * 1024

function Invoke-RadarPython {
    param([Parameter(Mandatory = $true)][string[]]$Arguments)
    $runtimeArguments = @('-X', 'utf8') + $Arguments
    $display = $python + ' ' + ($runtimeArguments -join ' ')
    Write-Host ('> ' + $display) -ForegroundColor Cyan
    $code = 1
    Push-Location -LiteralPath $srcRoot
    try {
        & $python @runtimeArguments | Out-Host
        $code = [int]$LASTEXITCODE
    }
    finally {
        Pop-Location
    }
    if ($code -ne 0) {
        throw ('Python command failed with exit code ' + $code + ': ' + $display)
    }
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

function Read-Utf8Json {
    param([Parameter(Mandatory = $true)][string]$Path)
    Assert-File -Path $Path -Description 'JSON artifact'
    return ConvertFrom-Json -InputObject ([System.IO.File]::ReadAllText($Path, [System.Text.UTF8Encoding]::new($false, $true)))
}

function Get-Sha256 {
    param([Parameter(Mandatory = $true)][string]$Path)
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Assert-Prepared {
    Assert-File -Path $contractPath -Description 'V26c contract'
    Assert-File -Path $readinessPath -Description 'V26c readiness'
    $contract = Read-Utf8Json -Path $contractPath
    $readiness = Read-Utf8Json -Path $readinessPath
    if ([string]$contract.schema_version -ne 'RADAR-LN3-V26c-l1-l2-known-only-oof-calibration-contract-v1') {
        throw 'V26c contract schema is invalid.'
    }
    if ([string]$readiness.schema_version -ne 'RADAR-LN3-V26c-l1-l2-known-only-oof-calibration-readiness-v1' -or
        [string]$readiness.status -ne 'passed' -or
        [string]$readiness.contract_sha256 -ne (Get-Sha256 -Path $contractPath) -or
        [int]$contract.unknown_training_exposure -ne 0 -or
        [int]$contract.unknown_normalization_exposure -ne 0 -or
        [int]$contract.unknown_oof_prediction_count -ne 0 -or
        [int]$contract.validation_access_count -ne 0 -or
        [int]$contract.test_access_count -ne 0) {
        throw 'V26c preparation artifacts violate the strict Known-only contract.'
    }
    $checks = @($readiness.checks.PSObject.Properties)
    if ($checks.Count -eq 0 -or @($checks | Where-Object { $_.Value -ne $true }).Count -ne 0) {
        throw 'V26c readiness has a failed or missing check.'
    }
}

function Assert-GpuMemoryHeadroom {
    $nvidiaSmi = Get-Command 'nvidia-smi.exe' -ErrorAction SilentlyContinue
    if ($null -eq $nvidiaSmi) {
        throw 'nvidia-smi.exe is unavailable; V26c full training requires CUDA.'
    }
    $output = @(& $nvidiaSmi.Source @('--id=0', '--query-gpu=memory.free', '--format=csv,noheader,nounits'))
    if ([int]$LASTEXITCODE -ne 0 -or $output.Count -ne 1) {
        throw 'Unable to query GPU 0 memory with nvidia-smi.'
    }
    $freeMiB = [int]::Parse(([string]$output[0]).Trim(), [System.Globalization.CultureInfo]::InvariantCulture)
    Write-Host ('[V26c] GPU0 free=' + $freeMiB + ' MiB; required=' + $minimumFreeGpuMemoryMiB) -ForegroundColor Cyan
    if ($freeMiB -lt $minimumFreeGpuMemoryMiB) {
        throw ('V26c CUDA preflight blocked: free GPU memory is below ' + $minimumFreeGpuMemoryMiB + ' MiB.')
    }
}

function Invoke-Prepare {
    Invoke-RadarPython @(
        '-m', 'py_compile',
        'build\known_gate_v26c_oof.py',
        'train\known_gate_v26c_oof.py',
        'report\known_gate_v26c_oof.py',
        'smoke\known_gate_v26c_oof_contract_test.py',
        'radar_ln3\known_gate_v26c_oof.py',
        'radar_ln3\known_gate.py',
        'radar_ln3\task_models.py',
        'radar_ln3\l1_l2_raw_routing.py',
        'radar_ln3\open_set_architecture.py'
    )
    Invoke-RadarPython @('-m', 'build.known_gate_v26c_oof', '--project-root', $projectPath)
    Assert-Prepared
}

function Invoke-Smoke {
    Assert-Prepared
    Invoke-RadarPython @('-m', 'smoke.known_gate_v26c_oof_contract_test', '--project-root', $projectPath)
    Invoke-RadarPython @('-m', 'smoke.static_isolation_check', '--root', $srcRoot)
}

function Get-FoldDirectory {
    param([Parameter(Mandatory = $true)][int]$FoldId)
    return Join-Path $artifactRoot ('fold-{0:D2}' -f $FoldId)
}

function Assert-FoldCompleted {
    param(
        [Parameter(Mandatory = $true)][int]$FoldId,
        [Parameter(Mandatory = $true)][ValidateSet('l1', 'l2')][string]$Task
    )
    $foldDirectory = Get-FoldDirectory -FoldId $FoldId
    $predictionPath = Join-Path $foldDirectory ($Task + '-held-out-predictions.json')
    $summaryPath = Join-Path $foldDirectory ($Task + '\training-summary.json')
    Assert-File -Path $predictionPath -Description ('V26c fold ' + $FoldId + ' ' + $Task + ' prediction')
    Assert-File -Path $summaryPath -Description ('V26c fold ' + $FoldId + ' ' + $Task + ' summary')
    $prediction = Read-Utf8Json -Path $predictionPath
    $summary = Read-Utf8Json -Path $summaryPath
    if ([string]$prediction.status -ne 'passed' -or [string]$summary.status -ne 'completed' -or
        [int]$summary.fixed_snapshot_epoch -ne $(if ($Task -eq 'l1') { 14 } else { 15 }) -or
        [int]$prediction.unknown_training_exposure -ne 0 -or
        [int]$prediction.validation_access_count -ne 0 -or
        [int]$prediction.test_payload_access_count -ne 0 -or
        [bool]$prediction.held_out_checkpoint_selection) {
        throw ('V26c fold ' + $FoldId + ' ' + $Task + ' output violates its fixed OOF contract.')
    }
    if ($Task -eq 'l2' -and (
        [bool]$prediction.held_out_l1_scores_reused_from_sealed_prediction -ne $true -or
        [int]$prediction.held_out_l1_forward_count -ne 0 -or
        [int]$prediction.held_out_l2_forward_count -ne 1 -or
        [int]$prediction.joint_l1_l2_scores.l2_held_out_forward_count -ne 1 -or
        [int]$prediction.joint_l1_l2_scores.l1_forward_count -ne 0 -or
        [bool]$prediction.joint_l1_l2_scores.l1_scores_reused_from_sealed_prediction -ne $true)) {
        throw ('V26c fold ' + $FoldId + ' L2 output violates the single held-out forward protocol.')
    }
}

function Invoke-FoldTraining {
    param(
        [Parameter(Mandatory = $true)][ValidateSet('l1', 'l2')][string]$Task,
        [Parameter(Mandatory = $true)][int]$FoldId
    )
    $foldDirectory = Get-FoldDirectory -FoldId $FoldId
    $predictionPath = Join-Path $foldDirectory ($Task + '-held-out-predictions.json')
    if (Test-Path -LiteralPath $predictionPath -PathType Leaf) {
        Write-Host ('[V26c] fold ' + $FoldId + ' ' + $Task + ' already completed; immutable output will be checked.') -ForegroundColor Yellow
        Assert-FoldCompleted -FoldId $FoldId -Task $Task
        return
    }
    $taskDirectory = Join-Path $foldDirectory $Task
    if (Test-Path -LiteralPath $taskDirectory -PathType Container) {
        $lastCheckpoint = Join-Path $taskDirectory 'last_model.pt'
        $entries = @(Get-ChildItem -LiteralPath $taskDirectory -Force)
        if ($entries.Count -gt 0 -and -not (Test-Path -LiteralPath $lastCheckpoint -PathType Leaf)) {
            throw ('V26c fold ' + $FoldId + ' ' + $Task + ' directory is non-empty without an epoch-boundary checkpoint; refusing to overwrite: ' + $taskDirectory)
        }
    }
    $arguments = @('-m', 'train.known_gate_v26c_oof', '--project-root', $projectPath, '--stage', $Task, '--fold-id', [string]$FoldId)
    $resume = Join-Path $foldDirectory ($Task + '\last_model.pt')
    if (Test-Path -LiteralPath $resume -PathType Leaf) {
        Write-Host ('[V26c] resuming fold ' + $FoldId + ' ' + $Task + ' from epoch-boundary checkpoint: ' + $resume) -ForegroundColor Yellow
        $arguments += @('--resume-checkpoint', $resume)
    }
    Invoke-RadarPython $arguments
    Assert-FoldCompleted -FoldId $FoldId -Task $Task
}

function Invoke-TrainingStage {
    param([Parameter(Mandatory = $true)][ValidateSet('l1', 'l2', 'train')][string]$TrainingStage)
    Assert-Prepared
    Assert-GpuMemoryHeadroom
    if ($TrainingStage -in @('l1', 'train')) {
        for ($foldId = 0; $foldId -lt 5; $foldId++) {
            Invoke-FoldTraining -Task 'l1' -FoldId $foldId
        }
    }
    if ($TrainingStage -in @('l2', 'train')) {
        for ($foldId = 0; $foldId -lt 5; $foldId++) {
            # L2 performs its joint L1->L2 held-out pass after epoch 15; the
            # corresponding L1 fixed snapshot must therefore already exist.
            Assert-FoldCompleted -FoldId $foldId -Task 'l1'
            Invoke-FoldTraining -Task 'l2' -FoldId $foldId
        }
    }
}

function Invoke-Calibrate {
    Assert-Prepared
    for ($foldId = 0; $foldId -lt 5; $foldId++) {
        Assert-FoldCompleted -FoldId $foldId -Task 'l1'
        Assert-FoldCompleted -FoldId $foldId -Task 'l2'
    }
    Invoke-RadarPython @('-m', 'report.known_gate_v26c_oof', '--project-root', $projectPath)
    Assert-File -Path $reportPath -Description 'V26c cross-fitted L3 report'
    Assert-File -Path $detailPath -Description 'V26c parent detail CSV'
    if ((Get-Item -LiteralPath $reportPath).Length -gt $maxReportBytes) {
        throw ('V26c report exceeds bounded size: ' + $reportPath)
    }
    $report = Read-Utf8Json -Path $reportPath
    if ([string]$report.status -ne 'completed_cross_fitted_known_only_l3_diagnostic' -or
        [bool]$report.deployment_eligible -or
        [int]$report.unknown_training_exposure -ne 0 -or
        [int]$report.unknown_threshold_fit_exposure -ne 0 -or
        [string]$report.mixed_unknown_audit -ne 'not_executed') {
        throw 'V26c report violates its Known-only diagnostic scope.'
    }
}

try {
    Assert-File -Path $python -Description 'Project Python interpreter'
    Write-Host ('[V26c] project=' + $projectPath) -ForegroundColor Green
    Write-Host ('[V26c] stage=' + $Stage + '; strict Known-only L1/L2 five-fold OOF plus cross-fitted L3 diagnostic') -ForegroundColor Green
    Write-Host '[V26c] Unknown is excluded from training, normalization, cache selection and threshold fitting; mixed Unknown audit remains unexecuted.' -ForegroundColor Green
    switch ($Stage) {
        'prepare' { Invoke-Prepare }
        'smoke' { Invoke-Smoke }
        'l1' { Invoke-TrainingStage -TrainingStage 'l1' }
        'l2' { Invoke-TrainingStage -TrainingStage 'l2' }
        'train' { Invoke-TrainingStage -TrainingStage 'train' }
        'calibrate' { Invoke-Calibrate }
        'all' {
            Invoke-Prepare
            Invoke-Smoke
            Invoke-TrainingStage -TrainingStage 'train'
            Invoke-Calibrate
        }
    }
    Write-Host '[V26c] requested stage completed.' -ForegroundColor Green
}
catch {
    [Console]::Error.WriteLine('[V26c] pipeline stopped: ' + $_.Exception.Message)
    exit 1
}
