[CmdletBinding()]
param(
    [ValidateSet('prepare', 'smoke', 'l1', 'train', 'calibrate', 'all')]
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
$contractPath = Join-Path $projectPath 'data\processed\RADAR-LN3-V26e-cq08-l1-epoch32-known-only-oof-calibration-contract.json'
$readinessPath = Join-Path $projectPath 'src-v3\processed\report\RADAR-LN3-V26e-l1-epoch32-known-only-oof-calibration-readiness.json'
$artifactRoot = Join-Path $projectPath 'artifacts\RADAR-LN3-V26e-cq08-l1-epoch32-known-only-oof-calibration'
$reportPath = Join-Path $projectPath 'src-v3\processed\report\RADAR-LN3-V26e-l1-epoch32-known-only-cross-fitted-l3-calibration.json'
$detailPath = Join-Path $projectPath 'src-v3\processed\report\RADAR-LN3-V26e-l1-epoch32-known-only-cross-fitted-l3-parent-details.csv'
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

function Get-ProjectRelativePath {
    param(
        [Parameter(Mandatory = $true)][string]$BasePath,
        [Parameter(Mandatory = $true)][string]$TargetPath
    )
    # Windows PowerShell 5.1/.NET Framework does not expose
    # [System.IO.Path]::GetRelativePath().  URI relative paths provide the
    # same deterministic project-relative representation on both runtimes.
    $baseFullPath = [System.IO.Path]::GetFullPath($BasePath).TrimEnd('\') + '\'
    $targetFullPath = [System.IO.Path]::GetFullPath($TargetPath)
    $baseUri = New-Object System.Uri($baseFullPath)
    $targetUri = New-Object System.Uri($targetFullPath)
    return [System.Uri]::UnescapeDataString($baseUri.MakeRelativeUri($targetUri).ToString()).Replace('\', '/')
}

function Assert-Prepared {
    Assert-File -Path $contractPath -Description 'V26e contract'
    Assert-File -Path $readinessPath -Description 'V26e readiness'
    $contract = Read-Utf8Json -Path $contractPath
    $readiness = Read-Utf8Json -Path $readinessPath
    if ([string]$contract.schema_version -ne 'RADAR-LN3-V26e-l1-epoch32-known-only-oof-calibration-contract-v1' -or
        [string]$readiness.schema_version -ne 'RADAR-LN3-V26e-l1-epoch32-known-only-oof-calibration-readiness-v1' -or
        [string]$readiness.status -ne 'passed' -or
        [string]$readiness.contract_sha256 -ne (Get-Sha256 -Path $contractPath) -or
        [int]$contract.unknown_training_exposure -ne 0 -or
        [int]$contract.unknown_normalization_exposure -ne 0 -or
        [int]$contract.unknown_oof_prediction_count -ne 0 -or
        [int]$contract.validation_access_count -ne 0 -or
        [int]$contract.test_access_count -ne 0 -or
        [int]$contract.l1.train_epoch_limit -ne 32 -or
        [int]$contract.l1.fixed_snapshot_epoch -ne 32 -or
        [int]$contract.l2.fixed_snapshot_epoch -ne 15 -or
        [bool]$contract.l2.training_executed_in_v26e) {
        throw 'V26e preparation artifacts violate the fixed epoch-32 L1 / reused epoch-15 L2 contract.'
    }
    $checks = @($readiness.checks.PSObject.Properties)
    if ($checks.Count -eq 0 -or @($checks | Where-Object { $_.Value -ne $true }).Count -ne 0) {
        throw 'V26e readiness has a failed or missing check.'
    }
}

function Assert-GpuMemoryHeadroom {
    $nvidiaSmi = Get-Command 'nvidia-smi.exe' -ErrorAction SilentlyContinue
    if ($null -eq $nvidiaSmi) {
        throw 'nvidia-smi.exe is unavailable; V26e full training requires CUDA.'
    }
    $output = @(& $nvidiaSmi.Source @('--id=0', '--query-gpu=memory.free', '--format=csv,noheader,nounits'))
    if ([int]$LASTEXITCODE -ne 0 -or $output.Count -ne 1) {
        throw 'Unable to query GPU 0 memory with nvidia-smi.'
    }
    $freeMiB = [int]::Parse(([string]$output[0]).Trim(), [System.Globalization.CultureInfo]::InvariantCulture)
    Write-Host ('[V26e] GPU0 free=' + $freeMiB + ' MiB; required=' + $minimumFreeGpuMemoryMiB) -ForegroundColor Cyan
    if ($freeMiB -lt $minimumFreeGpuMemoryMiB) {
        throw ('V26e CUDA preflight blocked: free GPU memory is below ' + $minimumFreeGpuMemoryMiB + ' MiB.')
    }
}

function Invoke-Prepare {
    Invoke-RadarPython @(
        '-m', 'py_compile',
        'build\known_gate_v26e_oof.py',
        'train\known_gate_v26e_oof.py',
        'report\known_gate_v26e_oof.py',
        'smoke\known_gate_v26e_oof_contract_test.py',
        'radar_ln3\known_gate_v26e_oof.py',
        'radar_ln3\known_gate_v26c_oof.py',
        'radar_ln3\known_gate.py',
        'radar_ln3\task_models.py',
        'radar_ln3\l1_l2_raw_routing.py',
        'radar_ln3\open_set_architecture.py'
    )
    Invoke-RadarPython @('-m', 'build.known_gate_v26e_oof', '--project-root', $projectPath)
    Assert-Prepared
}

function Invoke-Smoke {
    Assert-Prepared
    Invoke-RadarPython @('-m', 'smoke.known_gate_v26e_oof_contract_test', '--project-root', $projectPath)
    Invoke-RadarPython @('-m', 'smoke.static_isolation_check', '--root', $srcRoot)
}

function Get-FoldDirectory {
    param([Parameter(Mandatory = $true)][int]$FoldId)
    return Join-Path $artifactRoot ('fold-{0:D2}' -f $FoldId)
}

function Assert-FoldCompleted {
    param([Parameter(Mandatory = $true)][int]$FoldId)
    $foldDirectory = Get-FoldDirectory -FoldId $FoldId
    $predictionPath = Join-Path $foldDirectory 'l1-held-out-predictions.json'
    $summaryPath = Join-Path $foldDirectory 'l1\training-summary.json'
    $checkpointPath = Join-Path $foldDirectory 'l1\epoch-32-model.pt'
    $configPath = Join-Path $projectPath ('config\RADAR-LN3\RADAR-LN3-V26e-cq08-known-gate-oof-fold-{0:D2}-seed43-config.json' -f $FoldId)
    Assert-File -Path $predictionPath -Description ('V26e fold ' + $FoldId + ' L1 prediction')
    Assert-File -Path $summaryPath -Description ('V26e fold ' + $FoldId + ' L1 summary')
    Assert-File -Path $checkpointPath -Description ('V26e fold ' + $FoldId + ' L1 epoch-32 checkpoint')
    Assert-File -Path $configPath -Description ('V26e fold ' + $FoldId + ' L1 config')
    $prediction = Read-Utf8Json -Path $predictionPath
    $summary = Read-Utf8Json -Path $summaryPath
    $config = Read-Utf8Json -Path $configPath
    $contract = Read-Utf8Json -Path $contractPath
    $foldRows = @($contract.folds | Where-Object { [int]$_.fold_id -eq $FoldId })
    if ($foldRows.Count -ne 1) {
        throw ('V26e fold ' + $FoldId + ' is missing or repeated in its contract.')
    }
    $fold = $foldRows[0]
    $expectedTrainCount = [int]$fold.l1.train_parent_count
    $expectedHeldCount = [int]$fold.l1.held_out_parent_count
    $contractHash = Get-Sha256 -Path $contractPath
    $readinessHash = Get-Sha256 -Path $readinessPath
    $configHash = Get-Sha256 -Path $configPath
    $checkpointHash = Get-Sha256 -Path $checkpointPath
    $predictionHash = Get-Sha256 -Path $predictionPath
    $sourceConfigPath = Join-Path $projectPath 'config\RADAR-LN3\RADAR-LN3-V26-cq08-liquid-known-gate-known-only-open-set-baseline-seed43-config.json'
    $sourceConfigHash = Get-Sha256 -Path $sourceConfigPath
    $history = @($prediction.fold_training_history)
    $historyEpochs = @($history | ForEach-Object { [int]$_.epoch })
    $records = $prediction.records
    $recordScopes = @($records.PSObject.Properties.Name | Sort-Object)
    if ($recordScopes -join ',' -ne 'eligible_prefix,parent,window') {
        throw ('V26e fold ' + $FoldId + ' prediction scopes are incomplete.')
    }
    if ([string]$config.schema_version -ne 'RADAR-LN3-V26e-OOF-L1' -or
        ($config.PSObject.Properties.Name -contains 'v26c_oof') -or
        $null -ne $config.data.feature_cache_manifest_path -or
        $null -ne $config.data.expected_feature_cache_manifest_sha256 -or
        $null -ne $config.data.feature_experiment_contract_path -or
        $null -ne $config.data.expected_feature_experiment_contract_sha256 -or
        $null -ne $config.train.paired_baseline_config_path) {
        throw ('V26e fold ' + $FoldId + ' config retains obsolete V26/V21 identity fields.')
    }
    foreach ($scope in @('parent', 'eligible_prefix', 'window')) {
        $scopeRows = @($records.$scope)
        if ($scopeRows.Count -eq 0) {
            throw ('V26e fold ' + $FoldId + ' prediction scope is empty: ' + $scope)
        }
        $keys = @($scopeRows | ForEach-Object { [string]$_.parent_sample_id + '|' + [string]$_.event_index })
        if ($keys.Count -ne (@($keys | Sort-Object -Unique).Count) ) {
            throw ('V26e fold ' + $FoldId + ' prediction scope repeats a parent/event key: ' + $scope)
        }
    }
    $parentRows = @($records.parent)
    $parentIds = @($parentRows | ForEach-Object { [string]$_.parent_sample_id })
    $expectedHeldIds = @($fold.held_out_parent_ids | ForEach-Object { [string]$_ } | Sort-Object)
    if ($parentRows.Count -ne $expectedHeldCount -or
        $parentIds.Count -ne (@($parentIds | Sort-Object -Unique).Count) -or
        (@($parentIds | Sort-Object) -join '|') -ne ($expectedHeldIds -join '|')) {
        throw ('V26e fold ' + $FoldId + ' parent prediction coverage differs from the held-out fold.')
    }
    if ($history.Count -ne 32 -or (@($historyEpochs | Sort-Object) -join ',') -ne '1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32') {
        throw ('V26e fold ' + $FoldId + ' training history is not exactly epochs 1..32.')
    }
    foreach ($epochRecord in $history) {
        if ([int]$epochRecord.parent_order_count -ne $expectedTrainCount -or
            [int]$epochRecord.parent_order_unique_count -ne $expectedTrainCount -or
            [int]$epochRecord.held_out_forward_count -ne 0 -or
            [bool]$epochRecord.held_out_checkpoint_selection -or
            [int]$epochRecord.validation_access_count -ne 0 -or
            [int]$epochRecord.test_payload_access_count -ne 0) {
            throw ('V26e fold ' + $FoldId + ' training history violates its isolated epoch contract.')
        }
    }
    if ([string]$prediction.status -ne 'passed' -or
        [int]$prediction.fold_id -ne $FoldId -or
        [int]$summary.fold_id -ne $FoldId -or
        [string]$summary.status -ne 'completed' -or
        [int]$summary.fixed_snapshot_epoch -ne 32 -or
        [int]$prediction.fixed_snapshot_epoch -ne 32 -or
        [bool]$prediction.held_out_checkpoint_selection -or
        [int]$prediction.unknown_training_exposure -ne 0 -or
        [int]$prediction.validation_access_count -ne 0 -or
        [int]$prediction.test_payload_access_count -ne 0 -or
        [string]$prediction.config_path -ne (Get-ProjectRelativePath -BasePath $projectPath -TargetPath $configPath) -or
        [string]$prediction.config_sha256 -ne $configHash -or
        [string]$prediction.fold_contract_sha256 -ne $contractHash -or
        [string]$prediction.readiness_sha256 -ne $readinessHash -or
        [string]$prediction.source_config_sha256 -ne $sourceConfigHash -or
        [string]$prediction.checkpoint_path -ne (Get-ProjectRelativePath -BasePath $projectPath -TargetPath $checkpointPath) -or
        [string]$prediction.checkpoint_sha256 -ne $checkpointHash -or
        [string]$summary.prediction_path -ne (Get-ProjectRelativePath -BasePath $projectPath -TargetPath $predictionPath) -or
        [string]$summary.prediction_sha256 -ne $predictionHash -or
        [string]$summary.checkpoint_path -ne (Get-ProjectRelativePath -BasePath $projectPath -TargetPath $checkpointPath) -or
        [string]$summary.checkpoint_sha256 -ne $checkpointHash) {
        throw ('V26e fold ' + $FoldId + ' L1 output violates the fixed epoch-32 OOF contract.')
    }
}

function Invoke-FoldTraining {
    param([Parameter(Mandatory = $true)][int]$FoldId)
    $foldDirectory = Get-FoldDirectory -FoldId $FoldId
    $predictionPath = Join-Path $foldDirectory 'l1-held-out-predictions.json'
    if (Test-Path -LiteralPath $predictionPath -PathType Leaf) {
        Write-Host ('[V26e] fold ' + $FoldId + ' L1 already completed; immutable output will be checked.') -ForegroundColor Yellow
        Assert-FoldCompleted -FoldId $FoldId
        return
    }
    $taskDirectory = Join-Path $foldDirectory 'l1'
    if (Test-Path -LiteralPath $taskDirectory -PathType Container) {
        $lastCheckpoint = Join-Path $taskDirectory 'last_model.pt'
        $entries = @(Get-ChildItem -LiteralPath $taskDirectory -Force)
        if ($entries.Count -gt 0 -and -not (Test-Path -LiteralPath $lastCheckpoint -PathType Leaf)) {
            throw ('V26e fold ' + $FoldId + ' L1 directory is non-empty without an epoch-boundary checkpoint; refusing to overwrite: ' + $taskDirectory)
        }
    }
    $arguments = @('-m', 'train.known_gate_v26e_oof', '--project-root', $projectPath, '--fold-id', [string]$FoldId)
    $resume = Join-Path $taskDirectory 'last_model.pt'
    if (Test-Path -LiteralPath $resume -PathType Leaf) {
        Write-Host ('[V26e] resuming fold ' + $FoldId + ' L1 from epoch-boundary checkpoint: ' + $resume) -ForegroundColor Yellow
        $arguments += @('--resume-checkpoint', $resume)
    }
    Invoke-RadarPython $arguments
    Assert-FoldCompleted -FoldId $FoldId
}

function Invoke-TrainingStage {
    Assert-Prepared
    Assert-GpuMemoryHeadroom
    for ($foldId = 0; $foldId -lt 5; $foldId++) {
        Invoke-FoldTraining -FoldId $foldId
    }
}

function Invoke-Calibrate {
    Assert-Prepared
    for ($foldId = 0; $foldId -lt 5; $foldId++) {
        Assert-FoldCompleted -FoldId $foldId
    }
    Invoke-RadarPython @('-m', 'report.known_gate_v26e_oof', '--project-root', $projectPath)
    Assert-File -Path $reportPath -Description 'V26e cross-fitted L3 diagnostic'
    Assert-File -Path $detailPath -Description 'V26e parent detail CSV'
    if ((Get-Item -LiteralPath $reportPath).Length -gt $maxReportBytes) {
        throw ('V26e report exceeds bounded size: ' + $reportPath)
    }
    $report = Read-Utf8Json -Path $reportPath
    if ([string]$report.status -ne 'completed_v26e_fixed_epoch32_known_only_l3_diagnostic' -or
        [bool]$report.deployment_eligible -or
        [int]$report.unknown_training_exposure -ne 0 -or
        [int]$report.unknown_threshold_fit_exposure -ne 0 -or
        [string]$report.mixed_unknown_audit -ne 'not_executed') {
        throw 'V26e report violates its Known-only diagnostic scope.'
    }
}

try {
    Assert-File -Path $python -Description 'Project Python interpreter'
    Write-Host ('[V26e] project=' + $projectPath) -ForegroundColor Green
    Write-Host ('[V26e] stage=' + $Stage + '; fixed epoch-32 L1 Known-only five-fold OOF; V26c L2 epoch-15 read-only reuse') -ForegroundColor Green
    Write-Host '[V26e] Unknown, validation and test are excluded from training, normalization, checkpoint selection and threshold fitting.' -ForegroundColor Green
    switch ($Stage) {
        'prepare' { Invoke-Prepare }
        'smoke' { Invoke-Smoke }
        'l1' { Invoke-TrainingStage }
        'train' { Invoke-TrainingStage }
        'calibrate' { Invoke-Calibrate }
        'all' {
            Invoke-Prepare
            Invoke-Smoke
            Invoke-TrainingStage
            Invoke-Calibrate
        }
    }
    Write-Host '[V26e] requested stage completed.' -ForegroundColor Green
}
catch {
    [Console]::Error.WriteLine('[V26e] pipeline stopped: ' + $_.Exception.Message)
    exit 1
}
