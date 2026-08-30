[CmdletBinding()]
param(
    [ValidateSet('prepare', 'smoke', 'l1', 'train', 'calibrate', 'audit', 'all')]
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
$contractPath = Join-Path $projectPath 'data\processed\RADAR-LN3-V26j-cq08-l1-rd-causal-gru-epoch32-known-only-oof-calibration-contract.json'
$readinessPath = Join-Path $projectPath 'src-v3\processed\report\RADAR-LN3-V26j-l1-rd-causal-gru-epoch32-known-only-oof-calibration-readiness.json'
$artifactRoot = Join-Path $projectPath 'artifacts\RADAR-LN3-V26j-cq08-l1-rd-causal-gru-epoch32-known-only-oof-calibration'
$l3ReportPath = Join-Path $projectPath 'src-v3\processed\report\RADAR-LN3-V26j-l1-rd-causal-gru-epoch32-known-only-cross-fitted-l3-calibration.json'
$auditContractPath = Join-Path $projectPath 'data\processed\RADAR-LN3-V26j-cq08-frozen-mixed-open-set-audit-contract.json'
$auditReadinessPath = Join-Path $projectPath 'src-v3\processed\report\RADAR-LN3-V26j-frozen-mixed-open-set-audit-readiness.json'
$auditReportPath = Join-Path $projectPath 'src-v3\processed\report\RADAR-LN3-V26j-frozen-mixed-open-set-audit.json'
$minimumFreeGpuMemoryMiB = 4096

function Assert-File {
    param([Parameter(Mandatory=$true)][string]$Path,[Parameter(Mandatory=$true)][string]$Description)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { throw ($Description + ' is missing: ' + $Path) }
}

function Get-Sha256 { param([Parameter(Mandatory=$true)][string]$Path) return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant() }

function Read-Utf8Json {
    param([Parameter(Mandatory=$true)][string]$Path)
    Assert-File -Path $Path -Description 'JSON artifact'
    return ConvertFrom-Json -InputObject ([System.IO.File]::ReadAllText($Path, [System.Text.UTF8Encoding]::new($false, $true)))
}

function Invoke-RadarPython {
    param([Parameter(Mandatory=$true)][string[]]$Arguments)
    $runtimeArguments = @('-X','utf8') + $Arguments
    $display = $python + ' ' + ($runtimeArguments -join ' ')
    Write-Host ('> ' + $display) -ForegroundColor Cyan
    Push-Location -LiteralPath $srcRoot
    try { & $python @runtimeArguments | Out-Host; $code=[int]$LASTEXITCODE } finally { Pop-Location }
    if ($code -ne 0) { throw ('Python command failed with exit code ' + $code + ': ' + $display) }
}

function Assert-Prepared {
    Assert-File -Path $contractPath -Description 'V26j L1 contract'; Assert-File -Path $readinessPath -Description 'V26j L1 readiness'
    $contract=Read-Utf8Json $contractPath; $readiness=Read-Utf8Json $readinessPath; $checks=@($readiness.checks.PSObject.Properties)
    if ([string]$contract.schema_version -ne 'RADAR-LN3-V26j-l1-rd-causal-gru-epoch32-known-only-oof-calibration-contract-v1' -or [string]$readiness.schema_version -ne 'RADAR-LN3-V26j-l1-rd-causal-gru-epoch32-known-only-oof-calibration-readiness-v1' -or [string]$readiness.status -ne 'passed' -or [string]$readiness.contract_sha256 -ne (Get-Sha256 $contractPath) -or [string]$contract.single_variable.field -ne 'model.rd_frame_aggregation' -or [string]$contract.single_variable.treatment -ne 'causal_gru' -or [int]$contract.unknown_training_exposure -ne 0 -or [int]$contract.unknown_normalization_exposure -ne 0 -or [int]$contract.validation_access_count -ne 0 -or [int]$contract.test_access_count -ne 0 -or @($checks | Where-Object { $_.Value -ne $true }).Count -ne 0) { throw 'V26j preparation artifacts violate causal-GRU Known-only isolation.' }
}

function Assert-GpuMemoryHeadroom {
    $nvidiaSmi=Get-Command 'nvidia-smi.exe' -ErrorAction SilentlyContinue
    if ($null -eq $nvidiaSmi) { throw 'nvidia-smi.exe is unavailable; V26j training requires CUDA.' }
    $output=@(& $nvidiaSmi.Source @('--id=0','--query-gpu=memory.free','--format=csv,noheader,nounits'))
    if ([int]$LASTEXITCODE -ne 0 -or $output.Count -ne 1) { throw 'Unable to query GPU 0 memory.' }
    $freeMiB=[int]::Parse(([string]$output[0]).Trim(),[System.Globalization.CultureInfo]::InvariantCulture)
    Write-Host ('[V26j] GPU0 free=' + $freeMiB + ' MiB; required=' + $minimumFreeGpuMemoryMiB) -ForegroundColor Cyan
    if ($freeMiB -lt $minimumFreeGpuMemoryMiB) { throw 'V26j CUDA preflight blocked by insufficient free memory.' }
}

function Invoke-Prepare {
    Invoke-RadarPython @('-m','py_compile','radar_ln3\known_gate_features.py','radar_ln3\known_gate.py','radar_ln3\known_gate_v26j_oof.py','build\known_gate_v26j_oof.py','train\known_gate_v26j_oof.py','report\known_gate_v26j_oof.py','smoke\known_gate_v26j_oof_contract_test.py','radar_ln3\known_gate_v26j_audit.py','build\known_gate_v26j_audit.py','report\known_gate_v26j_audit.py','smoke\known_gate_v26j_audit_contract_test.py')
    Invoke-RadarPython @('-m','build.known_gate_v26j_oof','--project-root',$projectPath)
    Assert-Prepared
}

function Invoke-Smoke { Assert-Prepared; Invoke-RadarPython @('-m','smoke.known_gate_v26j_oof_contract_test','--project-root',$projectPath) }

function Get-FoldDirectory { param([Parameter(Mandatory=$true)][int]$FoldId) return (Join-Path $artifactRoot ('fold-{0:D2}' -f $FoldId)) }

function Assert-FoldCompleted {
    param([Parameter(Mandatory=$true)][int]$FoldId)
    $foldDir=Get-FoldDirectory $FoldId; $predictionPath=Join-Path $foldDir 'l1-held-out-predictions.json'; $summaryPath=Join-Path $foldDir 'l1\training-summary.json'; $checkpointPath=Join-Path $foldDir 'l1\epoch-32-model.pt'
    Assert-File $predictionPath ('V26j fold '+$FoldId+' prediction'); Assert-File $summaryPath ('V26j fold '+$FoldId+' summary'); Assert-File $checkpointPath ('V26j fold '+$FoldId+' checkpoint')
    $prediction=Read-Utf8Json $predictionPath; $summary=Read-Utf8Json $summaryPath; $runtime=$prediction.rd_runtime
    $history=@($prediction.fold_training_history); $expectedSteps=[int][Math]::Ceiling([double]$prediction.train_scope.parent_count / 2.0)
    $historyInvalid=($history.Count -ne 32)
    if (-not $historyInvalid) {
        for($epoch=1;$epoch -le 32;$epoch++) {
            $entry=$history[$epoch-1]
            if ([int]$entry.epoch -ne $epoch -or [int]$entry.global_step -ne ($epoch*$expectedSteps) -or [int]$entry.train_steps -ne $expectedSteps -or [int]$entry.parent_order_count -ne [int]$prediction.train_scope.parent_count -or [int]$entry.parent_order_unique_count -ne [int]$prediction.train_scope.parent_count -or [string]$entry.parent_order_sha256 -notmatch '^[0-9a-f]{64}$' -or [int]$entry.held_out_forward_count -ne 0 -or [bool]$entry.held_out_checkpoint_selection -or [int]$entry.validation_access_count -ne 0 -or [int]$entry.test_payload_access_count -ne 0 -or [string]$entry.rd_runtime.rd_frame_aggregation_runtime -ne 'causal_gru') { $historyInvalid=$true; break }
        }
    }
    if ([string]$prediction.schema_version -ne 'RADAR-LN3-V26j-l1-rd-causal-gru-epoch32-known-only-oof-prediction-v1' -or [int]$prediction.fold_id -ne $FoldId -or [int]$prediction.fixed_snapshot_epoch -ne 32 -or [string]$prediction.prediction_protocol -ne 'out_of_fold_fixed_epoch_32' -or [bool]$prediction.held_out_checkpoint_selection -or [int]$prediction.held_out_forward_count_before_fixed_snapshot -ne 0 -or [int]$prediction.held_out_prediction_pass_count -ne 1 -or [int]$prediction.unknown_training_exposure -ne 0 -or [int]$prediction.unknown_oof_prediction_count -ne 0 -or [int]$prediction.validation_access_count -ne 0 -or [int]$prediction.test_payload_access_count -ne 0 -or [string]$prediction.checkpoint_path -ne ('artifacts/RADAR-LN3-V26j-cq08-l1-rd-causal-gru-epoch32-known-only-oof-calibration/fold-{0:D2}/l1/epoch-32-model.pt' -f $FoldId) -or [string]$prediction.checkpoint_sha256 -ne (Get-Sha256 $checkpointPath) -or [string]$runtime.rd_frame_aggregation_config -ne 'causal_gru' -or [string]$runtime.rd_frame_aggregation_runtime -ne 'causal_gru' -or [string]$runtime.rd_temporal_module -ne 'GRUCell' -or [int]$runtime.rd_temporal_parameter_count -ne 3600 -or @($runtime.rd_temporal_state_keys).Count -ne 4 -or [string]$summary.status -ne 'completed' -or [int]$summary.fold_id -ne $FoldId -or [int]$summary.fixed_snapshot_epoch -ne 32 -or [int]$summary.global_step -ne (32*$expectedSteps) -or [string]$summary.prediction_path -ne ('artifacts/RADAR-LN3-V26j-cq08-l1-rd-causal-gru-epoch32-known-only-oof-calibration/fold-{0:D2}/l1-held-out-predictions.json' -f $FoldId) -or [string]$summary.prediction_sha256 -ne (Get-Sha256 $predictionPath) -or [string]$summary.checkpoint_sha256 -ne (Get-Sha256 $checkpointPath) -or [string]$summary.rd_runtime.rd_frame_aggregation_runtime -ne 'causal_gru' -or $historyInvalid) { throw ('V26j fold '+$FoldId+' output violates causal-GRU OOF contract.') }
    Invoke-RadarPython @('-m','train.known_gate_v26j_oof','--project-root',$projectPath,'--fold-id',[string]$FoldId,'--verify-fold')
}

function Invoke-FoldTraining {
    param([Parameter(Mandatory=$true)][int]$FoldId)
    $foldDir=Get-FoldDirectory $FoldId; $predictionPath=Join-Path $foldDir 'l1-held-out-predictions.json'
    if (Test-Path -LiteralPath $predictionPath -PathType Leaf) { Write-Host ('[V26j] fold '+$FoldId+' already completed; checking immutable output.') -ForegroundColor Yellow; Assert-FoldCompleted $FoldId; return }
    $taskDir=Join-Path $foldDir 'l1'; $resume=Join-Path $taskDir 'last_model.pt'
    if ((Test-Path -LiteralPath $taskDir -PathType Container) -and @(Get-ChildItem -LiteralPath $taskDir -Force).Count -gt 0 -and -not (Test-Path -LiteralPath $resume -PathType Leaf)) { throw ('V26j fold '+$FoldId+' non-empty without epoch-boundary checkpoint; refusing overwrite.') }
    $arguments=@('-m','train.known_gate_v26j_oof','--project-root',$projectPath,'--fold-id',[string]$FoldId)
    if (Test-Path -LiteralPath $resume -PathType Leaf) { $arguments += @('--resume-checkpoint',$resume) }
    Invoke-RadarPython $arguments; Assert-FoldCompleted $FoldId
}

function Invoke-Train { Assert-Prepared; Assert-GpuMemoryHeadroom; for($foldId=0;$foldId -lt 5;$foldId++){ Invoke-FoldTraining $foldId } }

function Invoke-Calibrate { Assert-Prepared; for($foldId=0;$foldId -lt 5;$foldId++){ Assert-FoldCompleted $foldId }; Invoke-RadarPython @('-m','report.known_gate_v26j_oof','--project-root',$projectPath); Assert-File $l3ReportPath 'V26j L3 report' }

function Invoke-Audit {
    Assert-File $l3ReportPath 'V26j frozen L3 report'
    Invoke-RadarPython @('-m','build.known_gate_v26j_audit','--project-root',$projectPath)
    Invoke-RadarPython @('-m','smoke.known_gate_v26j_audit_contract_test','--project-root',$projectPath)
    Invoke-RadarPython @('-m','report.known_gate_v26j_audit','--project-root',$projectPath)
    Assert-File $auditContractPath 'V26j audit contract'; Assert-File $auditReadinessPath 'V26j audit readiness'; Assert-File $auditReportPath 'V26j mixed audit report'
}

try {
    Assert-File $python 'Project Python interpreter'
    Write-Host ('[V26j] project='+$projectPath) -ForegroundColor Green
    Write-Host ('[V26j] stage='+$Stage+'; V26e-equivalent epoch32 Known-only OOF with only current-event RD masked_mean -> causal_gru') -ForegroundColor Green
    Write-Host '[V26j] Unknown is excluded from training, normalization, checkpoint selection and L3 threshold fitting.' -ForegroundColor Green
    switch($Stage) { 'prepare'{Invoke-Prepare}; 'smoke'{Invoke-Smoke}; 'l1'{Invoke-Train}; 'train'{Invoke-Train}; 'calibrate'{Invoke-Calibrate}; 'audit'{Invoke-Audit}; 'all'{Invoke-Prepare;Invoke-Smoke;Invoke-Train;Invoke-Calibrate;Invoke-Audit} }
    Write-Host '[V26j] requested stage completed.' -ForegroundColor Green
} catch { [Console]::Error.WriteLine('[V26j] pipeline stopped: '+$_.Exception.Message); exit 1 }
