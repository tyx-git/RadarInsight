#Requires -Version 5.1

[CmdletBinding()]
param(
    [switch]$PreflightOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Assert-Condition {
    param(
        [Parameter(Mandatory = $true)]
        [bool]$Condition,

        [Parameter(Mandatory = $true)]
        [string]$Message
    )

    if (-not $Condition) {
        throw $Message
    }
}

function Read-JsonFile {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    Assert-Condition -Condition (Test-Path -LiteralPath $Path -PathType Leaf) -Message "缺少JSON文件：$Path"
    try {
        return Get-Content -LiteralPath $Path -Raw -Encoding UTF8 | ConvertFrom-Json
    }
    catch {
        throw "JSON读取失败：$Path；$($_.Exception.Message)"
    }
}

function Get-Sha256 {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    Assert-Condition -Condition (Test-Path -LiteralPath $Path -PathType Leaf) -Message "缺少待校验文件：$Path"
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Assert-Sha256 {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,

        [Parameter(Mandatory = $true)]
        [string]$Expected
    )

    $actual = Get-Sha256 -Path $Path
    Assert-Condition -Condition ($actual -eq $Expected.ToLowerInvariant()) -Message "SHA-256不匹配：$Path；expected=$Expected；actual=$actual"
}

function Assert-StringArrayEqual {
    param(
        [Parameter(Mandatory = $true)]
        [object[]]$Actual,

        [Parameter(Mandatory = $true)]
        [object[]]$Expected,

        [Parameter(Mandatory = $true)]
        [string]$Scope
    )

    $actualStrings = @($Actual | ForEach-Object { [string]$_ } | Sort-Object)
    $expectedStrings = @($Expected | ForEach-Object { [string]$_ } | Sort-Object)
    Assert-Condition -Condition ($actualStrings.Count -eq $expectedStrings.Count) -Message "$Scope 数量不一致：actual=$($actualStrings.Count)，expected=$($expectedStrings.Count)"
    for ($index = 0; $index -lt $expectedStrings.Count; $index++) {
        Assert-Condition -Condition ($actualStrings[$index] -ceq $expectedStrings[$index]) -Message "$Scope 内容不一致：index=$index；actual=$($actualStrings[$index])；expected=$($expectedStrings[$index])"
    }
}

function Invoke-PythonStep {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Name,

        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )

    Write-Host ""
    Write-Host "========== $Name ==========" -ForegroundColor Cyan
    & $script:PythonExe @Arguments
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        throw "$Name 失败，Python退出码=$exitCode"
    }
}

function Get-FoldRecord {
    param(
        [Parameter(Mandatory = $true)]
        [object]$FoldContract,

        [Parameter(Mandatory = $true)]
        [int]$FoldId
    )

    $matches = @($FoldContract.folds | Where-Object { [int]$_.fold_id -eq $FoldId })
    Assert-Condition -Condition ($matches.Count -eq 1) -Message "fold contract中fold $FoldId 的记录数不是1"
    return $matches[0]
}

function Assert-OofFoldPrediction {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,

        [Parameter(Mandatory = $true)]
        [object]$Fold,

        [Parameter(Mandatory = $true)]
        [string]$AssignmentManifestSha256,

        [Parameter(Mandatory = $true)]
        [string]$FoldContractSha256
    )

    $payload = Read-JsonFile -Path $Path
    $foldId = [int]$Fold.fold_id
    $expectedHeldOutCount = [int]$Fold.held_out_parent_count
    $expectedTrainCount = [int]$Fold.train_parent_count

    Assert-Condition -Condition ([string]$payload.status -eq "passed") -Message "fold $foldId 状态不是passed"
    Assert-Condition -Condition ([string]$payload.schema_version -eq "RADAR-LN3-V14-known-only-parent-grouped-oof-prediction-v1") -Message "fold $foldId prediction schema不匹配"
    Assert-Condition -Condition ([int]$payload.fold_id -eq $foldId) -Message "fold $foldId prediction中的fold_id不匹配"
    Assert-Condition -Condition ([int]$payload.epochs -eq 20 -and [int]$payload.epoch -eq 20) -Message "fold $foldId 未达到固定20 epoch"
    Assert-Condition -Condition (-not [bool]$payload.early_stopping) -Message "fold $foldId 意外启用了early stopping"
    Assert-Condition -Condition (-not [bool]$payload.held_out_checkpoint_selection) -Message "fold $foldId 使用了held-out选择checkpoint"
    Assert-Condition -Condition ([int]$payload.held_out_forward_count_before_final_epoch -eq 0) -Message "fold $foldId 在最终epoch前访问了held-out"
    Assert-Condition -Condition ([int]$payload.held_out_prediction_pass_count -eq 1) -Message "fold $foldId held-out预测次数不是1"
    Assert-Condition -Condition ([int]$payload.unknown_training_exposure -eq 0 -and [int]$payload.unknown_oof_prediction_count -eq 0) -Message "fold $foldId 暴露了Unknown"
    Assert-Condition -Condition ([int]$payload.validation_access_count -eq 0 -and [int]$payload.test_payload_access_count -eq 0) -Message "fold $foldId 访问了validation或test"
    Assert-Condition -Condition ([string]$payload.assignment_manifest_sha256 -eq $AssignmentManifestSha256) -Message "fold $foldId assignment manifest hash不匹配"
    Assert-Condition -Condition ([string]$payload.fold_contract_sha256 -eq $FoldContractSha256) -Message "fold $foldId fold contract hash不匹配"
    Assert-Condition -Condition ([string]$payload.normalization_stats_sha256 -eq [string]$Fold.normalization_stats_sha256) -Message "fold $foldId normalization hash不匹配"
    Assert-Condition -Condition ([int]$payload.oof_prediction_parent_count -eq $expectedHeldOutCount) -Message "fold $foldId held-out parent数量不匹配"
    Assert-Condition -Condition ([string]$payload.checkpoint_set_id -match '^[0-9a-f]{32}$') -Message "fold $foldId checkpoint_set_id无效"

    Assert-Condition -Condition ([int]$payload.train_scope.parent_count -eq $expectedTrainCount) -Message "fold $foldId train scope数量不匹配"
    Assert-Condition -Condition ([int]$payload.train_scope.unknown_parent_count -eq 0 -and [int]$payload.train_scope.validation_parent_count -eq 0 -and [int]$payload.train_scope.test_parent_count -eq 0) -Message "fold $foldId train scope边界被污染"
    Assert-Condition -Condition ([string]$payload.train_scope.parent_ids_sha256 -eq [string]$Fold.train_parent_ids_sha256) -Message "fold $foldId train parent hash不匹配"
    Assert-Condition -Condition ([int]$payload.held_out_scope.parent_count -eq $expectedHeldOutCount) -Message "fold $foldId held-out scope数量不匹配"
    Assert-Condition -Condition ([int]$payload.held_out_scope.unknown_parent_count -eq 0 -and [int]$payload.held_out_scope.validation_parent_count -eq 0 -and [int]$payload.held_out_scope.test_parent_count -eq 0) -Message "fold $foldId held-out scope边界被污染"
    Assert-Condition -Condition ([string]$payload.held_out_scope.parent_ids_sha256 -eq [string]$Fold.held_out_parent_ids_sha256) -Message "fold $foldId held-out parent hash不匹配"

    $history = @($payload.fold_training_history)
    Assert-Condition -Condition ($history.Count -eq 20) -Message "fold $foldId history不是20个epoch"
    for ($historyIndex = 0; $historyIndex -lt 20; $historyIndex++) {
        $epochRecord = $history[$historyIndex]
        $expectedEpoch = $historyIndex + 1
        Assert-Condition -Condition ([int]$epochRecord.epoch -eq $expectedEpoch) -Message "fold $foldId history epoch不连续：expected=$expectedEpoch"
        Assert-Condition -Condition ([int]$epochRecord.parent_order_count -eq $expectedTrainCount -and [int]$epochRecord.parent_order_unique_count -eq $expectedTrainCount) -Message "fold $foldId epoch $expectedEpoch 未恰好消费每个train parent一次"
        Assert-Condition -Condition ([string]$epochRecord.train_parent_set_sha256 -eq [string]$Fold.train_parent_ids_sha256) -Message "fold $foldId epoch $expectedEpoch train parent集合不匹配"
        Assert-Condition -Condition ([int]$epochRecord.held_out_forward_count -eq 0 -and -not [bool]$epochRecord.held_out_checkpoint_selection) -Message "fold $foldId epoch $expectedEpoch 暴露了held-out"
        Assert-Condition -Condition ([int]$epochRecord.validation_access_count -eq 0 -and [int]$epochRecord.test_payload_access_count -eq 0) -Message "fold $foldId epoch $expectedEpoch 访问了validation或test"
    }

    $records = @($payload.records)
    Assert-Condition -Condition ($records.Count -eq $expectedHeldOutCount) -Message "fold $foldId records数量不匹配"
    $actualParentIds = @($records | ForEach-Object { [string]$_.parent_sample_id })
    Assert-Condition -Condition (@($actualParentIds | Sort-Object -Unique).Count -eq $actualParentIds.Count) -Message "fold $foldId records存在重复parent"
    Assert-Condition -Condition (@($records | Where-Object { [int]$_.true_id -ge 4 -or [string]$_.label_name -eq 'UnknownTarget' }).Count -eq 0) -Message "fold $foldId records包含Unknown或非known标签"
    Assert-StringArrayEqual -Actual $actualParentIds -Expected @($Fold.held_out_parent_ids) -Scope "fold $foldId held-out parent"

    foreach ($taskName in @("dbb_cu_gate", "dbb_specialist", "cu_specialist")) {
        $checkpointPath = [string]$payload.checkpoint_paths.$taskName
        $expectedCheckpointHash = [string]$payload.checkpoint_sha256.$taskName
        Assert-Condition -Condition (-not [string]::IsNullOrWhiteSpace($checkpointPath)) -Message "fold $foldId 缺少$taskName checkpoint路径"
        Assert-Condition -Condition ($expectedCheckpointHash -match '^[0-9a-f]{64}$') -Message "fold $foldId 缺少$taskName checkpoint hash"
        Assert-Sha256 -Path $checkpointPath -Expected $expectedCheckpointHash
    }

    Write-Host ("fold {0:D2} 校验通过：held-out={1}，epoch=20，Unknown/val/test exposure=0" -f $foldId, $records.Count) -ForegroundColor Green
    return $payload
}

function Assert-OofAggregate {
    param(
        [Parameter(Mandatory = $true)]
        [object[]]$FoldPayloads,

        [Parameter(Mandatory = $true)]
        [object]$AssignmentManifest
    )

    $allRecords = @($FoldPayloads | ForEach-Object { @($_.records) })
    $allParentIds = @($allRecords | ForEach-Object { [string]$_.parent_sample_id })
    $expectedParentIds = @($AssignmentManifest.assignments | ForEach-Object { [string]$_.parent_sample_id })
    Assert-Condition -Condition ($FoldPayloads.Count -eq 5) -Message "OOF完成折数不是5"
    Assert-Condition -Condition ($allRecords.Count -eq 1011) -Message "OOF记录总数不是1011：$($allRecords.Count)"
    Assert-Condition -Condition (@($allParentIds | Sort-Object -Unique).Count -eq 1011) -Message "OOF parent不是1011个唯一值"
    Assert-Condition -Condition (@($allRecords | Where-Object { [int]$_.true_id -ge 4 -or [string]$_.label_name -eq 'UnknownTarget' }).Count -eq 0) -Message "OOF汇总包含Unknown"
    Assert-StringArrayEqual -Actual $allParentIds -Expected $expectedParentIds -Scope "五折OOF完整parent集合"
    Write-Host "五折OOF汇总校验通过：1011个known train parent各恰好一次，Unknown/validation/test exposure=0" -ForegroundColor Green
}

function Assert-Calibration {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    $payload = Read-JsonFile -Path $Path
    $supportedSchemas = @(
        "RADAR-LN3-V14-strict-open-set-rejection-calibration-v1",
        "RADAR-LN3-V14-strict-open-set-rejection-calibration-v2"
    )
    Assert-Condition -Condition ($supportedSchemas -contains [string]$payload.schema_version) -Message "calibration schema不匹配"
    Assert-Condition -Condition ([string]$payload.source -eq "train_parent_grouped_oof") -Message "calibration来源不是train_parent_grouped_oof"
    Assert-Condition -Condition ([string]$payload.provenance.fit_split -eq "train" -and [string]$payload.provenance.fit_population -eq "known_only" -and [string]$payload.provenance.grouping -eq "parent" -and [string]$payload.provenance.prediction_protocol -eq "out_of_fold") -Message "calibration拟合范围不是known train parent-grouped OOF"
    Assert-Condition -Condition ([int]$payload.provenance.known_parent_count -eq 1011 -and [int]$payload.provenance.oof_prediction_parent_count -eq 1011 -and [int]$payload.provenance.oof_unique_parent_count -eq 1011) -Message "calibration OOF parent覆盖不完整"
    Assert-Condition -Condition ([int]$payload.provenance.unknown_training_exposure -eq 0 -and [int]$payload.provenance.unknown_threshold_fit_count -eq 0) -Message "calibration使用了Unknown"
    Assert-Condition -Condition (-not [bool]$payload.provenance.validation_used_for_threshold_fitting -and -not [bool]$payload.provenance.test_used) -Message "calibration使用了validation或test"
    Assert-Condition -Condition ([string]$payload.provenance.assignment_manifest_sha256 -eq $script:ExpectedAssignmentManifestSha256) -Message "calibration assignment manifest hash不匹配"
    Assert-Condition -Condition ([string]$payload.provenance.fold_contract_sha256 -eq $script:ExpectedFoldContractSha256) -Message "calibration fold contract hash不匹配"
    Assert-Condition -Condition (@($payload.provenance.oof_prediction_files).Count -eq 5) -Message "calibration provenance不是5折"

    foreach ($taskName in @("dbb_cu_gate", "dbb_specialist", "cu_specialist")) {
        $actualHash = [string]$payload.provenance.deployment_checkpoint_set.checkpoint_sha256.$taskName
        $expectedHash = [string]$script:FormalCheckpointHashes[$taskName]
        Assert-Condition -Condition ($actualHash -eq $expectedHash) -Message "calibration未绑定冻结的$taskName checkpoint"
        Assert-Condition -Condition ([string]$payload.provenance.deployment_checkpoint_set.task_config_identity_sha256.$taskName -match '^[0-9a-f]{64}$') -Message "calibration缺少$taskName config identity"
    }
    Assert-Condition -Condition ([int]$payload.provenance.deployment_checkpoint_set.epoch -gt 0 -and [int]$payload.provenance.deployment_checkpoint_set.global_step -gt 0) -Message "calibration部署epoch/global_step无效"
    Assert-Condition -Condition ([string]$payload.provenance.deployment_checkpoint_set.product_selection_sha256 -match '^[0-9a-f]{64}$') -Message "calibration缺少product selection hash"

    foreach ($fieldName in @("gate_accept_dbb_threshold", "dbb_accept_known_threshold", "cu_accept_clutter_threshold")) {
        $value = [double]$payload.$fieldName
        Assert-Condition -Condition (-not [double]::IsNaN($value) -and -not [double]::IsInfinity($value) -and $value -ge 0.0 -and $value -le 1.0) -Message "calibration字段${fieldName}不在[0,1]"
    }
    foreach ($fieldName in @("gate_accept_dbb_margin", "dbb_accept_known_margin", "cu_accept_clutter_margin")) {
        $value = [double]$payload.$fieldName
        Assert-Condition -Condition (-not [double]::IsNaN($value) -and -not [double]::IsInfinity($value) -and $value -ge -1.0 -and $value -le 1.0) -Message "calibration字段${fieldName}不在[-1,1]"
    }

    $componentTargetsPassed = [ordered]@{
        gate = [bool]$payload.selection_diagnostics.gate.all_precision_recall_targets_passed
        dbb_knownness = [bool]$payload.selection_diagnostics.dbb_knownness.all_precision_recall_targets_passed
        clutter_acceptance = [bool]$payload.selection_diagnostics.clutter_acceptance.all_precision_recall_targets_passed
    }
    $allTargetsPassed = -not ($componentTargetsPassed.Values -contains $false)
    $qualityStatus = if ($allTargetsPassed) { "deployment_eligible" } else { "audit_only" }
    $failureReasons = @(
        foreach ($entry in $componentTargetsPassed.GetEnumerator()) {
            if (-not [bool]$entry.Value) {
                "{0}_precision_recall_target_not_met" -f [string]$entry.Key
            }
        }
    )
    if ([string]$payload.schema_version -eq "RADAR-LN3-V14-strict-open-set-rejection-calibration-v2") {
        Assert-Condition -Condition ([string]$payload.calibration_quality_status -eq $qualityStatus) -Message "calibration质量状态与OOF诊断不一致"
        Assert-Condition -Condition ([bool]$payload.deployment_eligible -eq $allTargetsPassed) -Message "calibration部署资格与OOF诊断不一致"
        Assert-Condition -Condition ([bool]$payload.quality_gate.all_required_targets_passed -eq $allTargetsPassed) -Message "calibration质量门槛汇总不一致"
    }
    $payload | Add-Member -NotePropertyName calibration_quality_status -NotePropertyValue $qualityStatus -Force
    $payload | Add-Member -NotePropertyName deployment_eligible -NotePropertyValue $allTargetsPassed -Force
    $payload | Add-Member -NotePropertyName quality_gate_failure_reasons -NotePropertyValue $failureReasons -Force

    $qualityColor = if ($allTargetsPassed) { "Green" } else { "Yellow" }
    Write-Host "OOF calibration来源/绑定校验通过；质量状态=$qualityStatus，deployment_eligible=$allTargetsPassed" -ForegroundColor $qualityColor
    return $payload
}

function Assert-FrozenValidationReport {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,

        [Parameter(Mandatory = $true)]
        [string]$CalibrationPath
    )

    $payload = Read-JsonFile -Path $Path
    $calibration = Assert-Calibration -Path $CalibrationPath
    $expectedReportStatus = if ([bool]$calibration.deployment_eligible) { "passed" } else { "audit_only" }
    $hasDeploymentEligible = $null -ne $payload.PSObject.Properties["deployment_eligible"]
    $legacyAuditReport = (
        -not [bool]$calibration.deployment_eligible -and
        [string]$payload.status -eq "passed" -and
        -not $hasDeploymentEligible
    )
    Assert-Condition -Condition ([string]$payload.status -eq $expectedReportStatus -or $legacyAuditReport) -Message "冻结development validation报告质量状态不匹配"
    Assert-Condition -Condition ([string]$payload.schema_version -like 'RADAR-LN3-V14*') -Message "冻结development validation不是V14 schema"
    Assert-Condition -Condition ([string]$payload.split -eq "val") -Message "冻结development validation报告不是val split"
    Assert-Condition -Condition ([int]$payload.dataset_parent_count -eq 232 -and [int]$payload.known_parent_count -eq 217 -and [int]$payload.unknown_audit_parent_count -eq 15) -Message "冻结development validation parent范围不匹配"
    Assert-Condition -Condition (-not [bool]$payload.product_semantics.unknown_is_model_class -and [int]$payload.product_semantics.unknown_training_exposure -eq 0) -Message "冻结development validation错误地把Unknown当作模型类"
    Assert-Condition -Condition ([string]$payload.development_unknown_audit.status -eq "frozen_development_open_set_audit_only") -Message "Unknown评估不是冻结development audit"
    Assert-Condition -Condition (-not [bool]$payload.development_unknown_audit.threshold_fitting_allowed -and -not [bool]$payload.development_unknown_audit.checkpoint_selection_allowed -and -not [bool]$payload.development_unknown_audit.unknown_is_learned_class) -Message "development Unknown被用于阈值、checkpoint或学习"
    Assert-Condition -Condition ([string]$payload.checkpoint_selection_inputs.status -eq "forbidden_for_development_open_set_audit" -and -not [bool]$payload.checkpoint_selection_inputs.unknown_used_for_selection) -Message "冻结development audit意外参与checkpoint选择"
    Assert-Condition -Condition ([string]$payload.hierarchical_calibration.source -eq "train_parent_grouped_oof") -Message "冻结development validation未使用OOF calibration"
    Assert-Condition -Condition ([System.IO.Path]::GetFullPath([string]$payload.hierarchical_calibration.path) -eq [System.IO.Path]::GetFullPath($CalibrationPath)) -Message "冻结development validation使用了错误calibration文件"
    Assert-Condition -Condition ([string]$payload.calibrated_hierarchical_action.status -eq "evaluated_with_train_parent_grouped_oof_calibration") -Message "五类开放集动作未完成评估"
    Assert-Condition -Condition ([int]$payload.calibrated_hierarchical_action.known_support -eq 217 -and [int]$payload.calibrated_hierarchical_action.unknown_support -eq 15) -Message "五类开放集动作评估分母不匹配"
    Assert-Condition -Condition ($null -ne $payload.calibrated_hierarchical_action.action_metrics_reject_mapped_to_unknown_id) -Message "缺少四known接受类别加UNKNOWN_REJECTED的五动作指标"
    Assert-Condition -Condition ([string]$payload.calibrated_hierarchical_action.action_mapping_warning -like '*UNKNOWN_REJECTED*') -Message "报告未区分Unknown拒识动作与普通闭集类别"
    $formalRejectionStates = @(
        "executed_with_calibration",
        "audit_only_evaluated_with_calibration",
        "executed_with_deployment_eligible_calibration"
    )
    Assert-Condition -Condition ([string]$payload.full_validation -eq "executed" -and $formalRejectionStates -contains [string]$payload.formal_rejection) -Message "冻结development validation执行状态不完整"
    Assert-Condition -Condition ([string]$payload.test_evaluation -eq "未执行" -and [string]$payload.p95 -eq "未执行") -Message "脚本越权执行了test或P95"

    foreach ($taskName in @("dbb_cu_gate", "dbb_specialist", "cu_specialist")) {
        $actualCheckpoint = [System.IO.Path]::GetFullPath([string]$payload.checkpoints.$taskName)
        $expectedCheckpoint = [System.IO.Path]::GetFullPath([string]$script:FormalCheckpointPaths[$taskName])
        Assert-Condition -Condition ($actualCheckpoint -eq $expectedCheckpoint) -Message "冻结development validation使用了错误的$taskName checkpoint"
    }
    Assert-Condition -Condition (Test-Path -LiteralPath ([string]$payload.sample_detail_information) -PathType Leaf) -Message "冻结development validation缺少sample detail"

    $payload | Add-Member -NotePropertyName effective_calibration_quality_status -NotePropertyValue ([string]$calibration.calibration_quality_status) -Force
    $payload | Add-Member -NotePropertyName effective_deployment_eligible -NotePropertyValue ([bool]$calibration.deployment_eligible) -Force
    $reportColor = if ([bool]$calibration.deployment_eligible) { "Green" } else { "Yellow" }
    Write-Host "冻结development validation执行校验通过：217 known + 15 Unknown；质量状态=$($calibration.calibration_quality_status)，非部署放行" -ForegroundColor $reportColor
    return $payload
}

$ProjectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
$SourceRoot = Join-Path $ProjectRoot "src-v3"
$PythonExe = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$ConfigRoot = Join-Path $ProjectRoot "config\RADAR-LN3"
$FormalArtifactRoot = Join-Path $ProjectRoot "artifacts\RADAR-LN3-V14-cq08-liquid-strict-open-set-cascade-formal"
$OofArtifactRoot = Join-Path $ProjectRoot "artifacts\RADAR-LN3-V14-cq08-liquid-strict-open-set-parent-grouped-oof"
$AssignmentManifestPath = Join-Path $ProjectRoot "data\processed\RADAR-LN3-V14-cq08-known-only-parent-grouped-oof-manifest.json"
$FoldContractPath = Join-Path $ProjectRoot "data\processed\RADAR-LN3-V14-cq08-known-only-parent-grouped-oof-fold-contract.json"
$CalibrationPath = Join-Path $FormalArtifactRoot "strict-open-set-rejection-calibration.json"
$ValidationOutputDir = Join-Path $SourceRoot "processed\report\RADAR-LN3-V14-strict-open-set-calibrated-validation"
$ValidationReportPath = Join-Path $ValidationOutputDir "evaluation-val.json"

$GateConfig = Join-Path $ConfigRoot "RADAR-LN3-V14-cq08-liquid-strict-open-set-dbb-cu-gate-config.json"
$DbbConfig = Join-Path $ConfigRoot "RADAR-LN3-V14-cq08-liquid-strict-open-set-dbb-specialist-config.json"
$CuConfig = Join-Path $ConfigRoot "RADAR-LN3-V14-cq08-liquid-strict-open-set-cu-specialist-config.json"
$GateCheckpoint = Join-Path $FormalArtifactRoot "dbb_cu_gate\best_model.pt"
$DbbCheckpoint = Join-Path $FormalArtifactRoot "dbb_specialist\best_model.pt"
$CuCheckpoint = Join-Path $FormalArtifactRoot "cu_specialist\best_model.pt"

$ExpectedAssignmentManifestSha256 = "1aea914c9566a8495471590ea5ebf9d6b62da6a6788b25aa7b39c5e6ebf7a46f"
$ExpectedFoldContractSha256 = "8a6404fb7933bb8de29d561986494fde628f15c998c92fedd8b6b6b1088ce559"
$FormalCheckpointPaths = @{
    dbb_cu_gate = $GateCheckpoint
    dbb_specialist = $DbbCheckpoint
    cu_specialist = $CuCheckpoint
}
$FormalCheckpointHashes = @{
    dbb_cu_gate = "b6fcff98e282709f0fb10d9d2ab447615436635f3b8e82d762a302f490ce462b"
    dbb_specialist = "b60a2e35c150866073c5d652be93d580df144939182c6b5eff652232d1d33687"
    cu_specialist = "e8ba8ba540a66759631b93cd6177a6d14602b9f6f7122b674e5477695a3da0c2"
}

$lockStream = $null
$locationPushed = $false
try {
    Write-Host "========== V14严格开放集OOF流水线预检 ==========" -ForegroundColor Cyan
    Assert-Condition -Condition (Test-Path -LiteralPath $PythonExe -PathType Leaf) -Message "缺少项目Python解释器：$PythonExe"
    Assert-Condition -Condition (Test-Path -LiteralPath $SourceRoot -PathType Container) -Message "缺少src-v3目录：$SourceRoot"

    Assert-Sha256 -Path $GateConfig -Expected "9d1d708cc3955c1584357a3ea5207b1076d9ad3c59cd8b99d89648c7b0faa8b3"
    Assert-Sha256 -Path $DbbConfig -Expected "23d2f158b368003b3b1c927c6cd7b14cd10d86e09d5c1d431135d571e2ac2da0"
    Assert-Sha256 -Path $CuConfig -Expected "3beddc09d60a47edf15276d2be92b8345d0a9fb2e99fdcf1c9643f99252e3af6"
    Assert-Sha256 -Path $AssignmentManifestPath -Expected $ExpectedAssignmentManifestSha256
    Assert-Sha256 -Path $FoldContractPath -Expected $ExpectedFoldContractSha256
    foreach ($taskName in @("dbb_cu_gate", "dbb_specialist", "cu_specialist")) {
        Assert-Sha256 -Path ([string]$FormalCheckpointPaths[$taskName]) -Expected ([string]$FormalCheckpointHashes[$taskName])
    }

    $assignmentManifest = Read-JsonFile -Path $AssignmentManifestPath
    $foldContract = Read-JsonFile -Path $FoldContractPath
    Assert-Condition -Condition ([int]$assignmentManifest.fold_count -eq 5 -and [int]$assignmentManifest.known_parent_count -eq 1011 -and [int]$assignmentManifest.unknown_oof_parent_count -eq 0 -and [int]$assignmentManifest.validation_parent_count -eq 0 -and [int]$assignmentManifest.test_parent_count -eq 0) -Message "assignment manifest范围不符合冻结的5折known-only OOF契约"
    Assert-Condition -Condition ([int]$foldContract.fold_count -eq 5 -and [int]$foldContract.known_parent_count -eq 1011 -and [int]$foldContract.unknown_training_exposure -eq 0 -and [int]$foldContract.unknown_oof_prediction_count -eq 0 -and [int]$foldContract.validation_access_count -eq 0 -and [int]$foldContract.test_payload_access_count -eq 0) -Message "fold contract范围不符合冻结的known-only OOF契约"

    if (-not (Test-Path -LiteralPath $OofArtifactRoot -PathType Container)) {
        New-Item -ItemType Directory -Path $OofArtifactRoot | Out-Null
    }
    $lockPath = Join-Path $OofArtifactRoot ".v14-oof-pipeline.lock"
    try {
        $lockStream = [System.IO.File]::Open(
            $lockPath,
            [System.IO.FileMode]::OpenOrCreate,
            [System.IO.FileAccess]::ReadWrite,
            [System.IO.FileShare]::None
        )
    }
    catch {
        throw "无法获得OOF流水线独占锁；可能已有同一脚本正在运行：$lockPath"
    }

    Push-Location -LiteralPath $SourceRoot
    $locationPushed = $true
    Write-Host "预检通过。不会访问test，不会执行P95，不会使用Unknown拟合阈值。" -ForegroundColor Green

    if ($PreflightOnly) {
        $completedFoldCount = 0
        $completedFoldPayloads = @()
        for ($foldId = 0; $foldId -lt 5; $foldId++) {
            $fold = Get-FoldRecord -FoldContract $foldContract -FoldId $foldId
            $foldRoot = Join-Path $OofArtifactRoot ("fold-{0:D2}" -f $foldId)
            $predictionPath = Join-Path $foldRoot "held-out-predictions.json"
            $resumeCheckpoint = Join-Path $foldRoot "joint_last_model.pt"
            if (Test-Path -LiteralPath $predictionPath -PathType Leaf) {
                $completedFoldPayloads += ,(Assert-OofFoldPrediction -Path $predictionPath -Fold $fold -AssignmentManifestSha256 $ExpectedAssignmentManifestSha256 -FoldContractSha256 $ExpectedFoldContractSha256)
                $completedFoldCount++
            }
            elseif (Test-Path -LiteralPath $resumeCheckpoint -PathType Leaf) {
                Write-Host ("fold {0:D2}：未完成，存在可恢复checkpoint：{1}" -f $foldId, $resumeCheckpoint) -ForegroundColor Yellow
            }
            elseif (Test-Path -LiteralPath $foldRoot -PathType Container) {
                $existingEntries = @(Get-ChildItem -LiteralPath $foldRoot -Force)
                Assert-Condition -Condition ($existingEntries.Count -eq 0) -Message "fold $foldId 目录非空但缺少可恢复joint_last_model.pt；为防覆盖已停止：$foldRoot"
                Write-Host ("fold {0:D2}：尚未启动，将从epoch 1开始。" -f $foldId) -ForegroundColor Yellow
            }
            else {
                Write-Host ("fold {0:D2}：尚未启动，将从epoch 1开始。" -f $foldId) -ForegroundColor Yellow
            }
        }
        if ($completedFoldCount -eq 5) {
            Assert-OofAggregate -FoldPayloads $completedFoldPayloads -AssignmentManifest $assignmentManifest
        }
        if (Test-Path -LiteralPath $CalibrationPath -PathType Leaf) {
            Assert-Calibration -Path $CalibrationPath | Out-Null
        }
        if (Test-Path -LiteralPath $ValidationReportPath -PathType Leaf) {
            Assert-FrozenValidationReport -Path $ValidationReportPath -CalibrationPath $CalibrationPath | Out-Null
        }
        Write-Host ""
        Write-Host "预检模式完成：已完成并验证 $completedFoldCount/5 折；仅校验已有calibration/report，未启动训练、校准或validation评估。" -ForegroundColor Green
        return
    }

    $foldPayloads = @()
    for ($foldId = 0; $foldId -lt 5; $foldId++) {
        $fold = Get-FoldRecord -FoldContract $foldContract -FoldId $foldId
        $foldRoot = Join-Path $OofArtifactRoot ("fold-{0:D2}" -f $foldId)
        $predictionPath = Join-Path $foldRoot "held-out-predictions.json"

        if (Test-Path -LiteralPath $predictionPath -PathType Leaf) {
            Write-Host ""
            Write-Host ("fold {0:D2} 已完成，执行严格只读校验后跳过训练。" -f $foldId) -ForegroundColor Yellow
        }
        else {
            $arguments = @(
                "-m", "train.strict_open_set_oof",
                "--project-root", $ProjectRoot,
                "--fold-id", [string]$foldId,
                "--gate-config", $GateConfig,
                "--dbb-config", $DbbConfig,
                "--cu-config", $CuConfig,
                "--prediction-batch-size", "2"
            )
            $resumeCheckpoint = Join-Path $foldRoot "joint_last_model.pt"
            if (Test-Path -LiteralPath $resumeCheckpoint -PathType Leaf) {
                Write-Host ("fold {0:D2} 从最后一个完整joint checkpoint恢复：{1}" -f $foldId, $resumeCheckpoint) -ForegroundColor Yellow
                $arguments += @("--resume-checkpoint", $resumeCheckpoint)
            }
            elseif (Test-Path -LiteralPath $foldRoot -PathType Container) {
                $existingEntries = @(Get-ChildItem -LiteralPath $foldRoot -Force)
                Assert-Condition -Condition ($existingEntries.Count -eq 0) -Message "fold $foldId 目录非空但缺少可恢复joint_last_model.pt；为防覆盖已停止：$foldRoot"
            }
            Invoke-PythonStep -Name ("训练并预测OOF fold {0:D2}" -f $foldId) -Arguments $arguments
        }

        $foldPayloads += ,(Assert-OofFoldPrediction -Path $predictionPath -Fold $fold -AssignmentManifestSha256 $ExpectedAssignmentManifestSha256 -FoldContractSha256 $ExpectedFoldContractSha256)
    }

    Assert-OofAggregate -FoldPayloads $foldPayloads -AssignmentManifest $assignmentManifest

    Invoke-PythonStep -Name "仅用known train parent-grouped OOF拟合并冻结三道接受边界" -Arguments @(
        "-m", "report.strict_open_set_oof",
        "--project-root", $ProjectRoot,
        "--gate-config", $GateConfig,
        "--dbb-config", $DbbConfig,
        "--cu-config", $CuConfig,
        "--gate-checkpoint", $GateCheckpoint,
        "--dbb-checkpoint", $DbbCheckpoint,
        "--cu-checkpoint", $CuCheckpoint,
        "--output", $CalibrationPath
    )
    $calibration = Assert-Calibration -Path $CalibrationPath

    if (Test-Path -LiteralPath $ValidationReportPath -PathType Leaf) {
        Write-Host ""
        Write-Host "冻结development validation报告已存在；只读校验后跳过重复评估。" -ForegroundColor Yellow
    }
    else {
        Invoke-PythonStep -Name "使用冻结calibration执行一次development validation五类开放集评估" -Arguments @(
            "-m", "report.cascade",
            "--project-root", $ProjectRoot,
            "--gate-config", $GateConfig,
            "--dbb-config", $DbbConfig,
            "--cu-config", $CuConfig,
            "--gate-checkpoint", $GateCheckpoint,
            "--dbb-checkpoint", $DbbCheckpoint,
            "--cu-checkpoint", $CuCheckpoint,
            "--split", "val",
            "--batch-size", "8",
            "--output-dir", $ValidationOutputDir,
            "--calibration", $CalibrationPath,
            "--calibration-purpose", "audit"
        )
    }
    $validation = Assert-FrozenValidationReport -Path $ValidationReportPath -CalibrationPath $CalibrationPath

    Write-Host ""
    $pipelineColor = if ([bool]$calibration.deployment_eligible) { "Green" } else { "Yellow" }
    Write-Host "========== V14严格开放集流水线执行完成；质量状态=$($calibration.calibration_quality_status) ==========" -ForegroundColor $pipelineColor
    [ordered]@{
        status = [string]$calibration.calibration_quality_status
        execution_status = "passed"
        deployment_eligible = [bool]$calibration.deployment_eligible
        quality_gate_failure_reasons = @($calibration.quality_gate_failure_reasons)
        oof_fold_count = 5
        oof_known_parent_count = 1011
        unknown_oof_or_threshold_fit_count = 0
        calibration = $CalibrationPath
        calibration_sha256 = Get-Sha256 -Path $CalibrationPath
        development_validation_report = $ValidationReportPath
        development_validation_parent_count = [int]$validation.dataset_parent_count
        open_set_action_semantics = "Drone/Bird/Balloon/Clutter accepted labels plus UNKNOWN_REJECTED action; Unknown is not a learned class"
        test = "未执行"
        p95 = "未执行"
        deployment = "未执行"
    } | ConvertTo-Json -Depth 4
}
catch {
    Write-Error "V14严格开放集流水线已停止：$($_.Exception.Message)"
    exit 1
}
finally {
    if ($locationPushed) {
        Pop-Location
    }
    if ($null -ne $lockStream) {
        $lockStream.Dispose()
    }
}
