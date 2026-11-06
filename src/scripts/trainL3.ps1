[CmdletBinding()]
param(
    [string]$Root = "artifacts/radar_ln3",
    [ValidateSet("all", "cache", "heads", "calibration", "audit")]
    [string]$Stage = "all",
    [ValidateSet("cuda", "cpu")]
    [string]$Device,
    [ValidateRange(1, 64)]
    [int]$BatchSize,
    [ValidateRange(1, 256)]
    [int]$HeadBatchSize,
    [ValidateRange(1, 10000)]
    [int]$Epochs,
    [ValidateRange(1, 10000)]
    [int]$ProgressInterval = 25,
    [double]$LearningRate,
    [double]$WeightDecay,
    [switch]$Overwrite,
    [switch]$Resume,
    [switch]$ResetState,
    [switch]$SkipDataValidation,
    [string]$PythonPath = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ($Resume -and ($ResetState -or $Overwrite)) {
    throw "Resume 不能与 ResetState 或 Overwrite 同时使用。"
}

$scriptDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = (Resolve-Path (Join-Path $scriptDirectory "..\..")).Path
. (Join-Path $scriptDirectory "common\pipeline_inputs.ps1")
$rootPath = if ([IO.Path]::IsPathRooted($Root)) { [IO.Path]::GetFullPath($Root) } else { [IO.Path]::GetFullPath((Join-Path $projectRoot $Root)) }
$rootRelative = [IO.Path]::GetRelativePath($projectRoot, $rootPath)
if ($rootRelative -eq ".." -or $rootRelative.StartsWith("..$([IO.Path]::DirectorySeparatorChar)") -or [IO.Path]::IsPathRooted($rootRelative)) { throw "Root 必须位于项目目录内：$Root" }
$pipelineRoot = Join-Path $rootPath "pipeline\l3"
$headRoot = "$($rootRelative.Replace('\','/'))/l3/heads"
$representationRoot = "$($rootRelative.Replace('\','/'))/l3/representations"
$reportRoot = "$($rootRelative.Replace('\','/'))/l3"
$manifestPath = "src/config/manifests.json"
$modelConfigPath = "src/config/model.json"
$l1ConfigPath = "src/config/model.json"
$l2ConfigPath = "src/config/model.json"
$l1CheckpointPath = "$($rootRelative.Replace('\','/'))/l1/l1.pt"
$l2CheckpointPath = "$($rootRelative.Replace('\','/'))/l2/l2.pt"
$l1HeadPath = "$headRoot/l1-knownness.pt"
$l2HeadPath = "$headRoot/l2-knownness.pt"
$decisionProfilePath = "src/config/radar_ln3_decision.json"
$policyPath = "src/config/radar_ln3_policy.json"
$l3ConfigPath = "src/config/radar_ln3_l3.json"
$cacheRoot = "data/cq08/cache/features/l3"
$headFitCache = "$cacheRoot/head_fit/feature-cache-manifest.json"
$validationCache = "$cacheRoot/validation/feature-cache-manifest.json"
$auditManifest = "src/config/radar_ln3_audit_manifest.json"
$auditCache = "$cacheRoot/audit/feature-cache-manifest.json"
$calibrationPath = "$reportRoot/radar_ln3_calibration.exploratory.json"
$auditedCalibrationPath = "$reportRoot/radar_ln3_calibration.json"
$auditReportPath = "$reportRoot/radar_ln3_audit_report.json"
$statePath = Join-Path $pipelineRoot "state.json"
$logRoot = Join-Path $pipelineRoot "logs"
$commandRoot = Join-Path $pipelineRoot "commands"
$lockPath = Join-Path $pipelineRoot "run.lock"

# 正式训练参数从同一份配置读取，显式命令行参数优先。
$l3Config = Get-Content -LiteralPath (Join-Path $projectRoot $l3ConfigPath) -Raw -Encoding UTF8 | ConvertFrom-Json
$knownnessTraining = $l3Config.knownness_training
foreach ($entry in @{Device="device"; BatchSize="batch_size"; HeadBatchSize="head_batch_size"; Epochs="epochs"; LearningRate="learning_rate"; WeightDecay="weight_decay"}.GetEnumerator()) {
    if (-not $PSBoundParameters.ContainsKey($entry.Key)) {
        Set-Variable -Name $entry.Key -Value $knownnessTraining.common.($entry.Value)
    }
    $knownnessTraining.common.($entry.Value) = Get-Variable -Name $entry.Key -ValueOnly
}
$knownnessTrainingJson = $knownnessTraining | ConvertTo-Json -Depth 10 -Compress

if ([string]::IsNullOrWhiteSpace($PythonPath)) {
    $PythonPath = Join-Path $projectRoot ".venv\Scripts\python.exe"
} else {
    $PythonPath = (Resolve-Path -LiteralPath $PythonPath).Path
}
if (-not (Test-Path -LiteralPath $PythonPath -PathType Leaf)) {
    throw "Python 解释器不存在：$PythonPath"
}

New-Item -ItemType Directory -Force -Path $pipelineRoot, $logRoot, $commandRoot, $reportRoot | Out-Null
$rootPreparationArgs = @(
    "-B", "src/scripts/prepare_artifact_root.py", "--project-root", $projectRoot,
    "--root", $rootPath, "--scope", "l3"
)
if ($Overwrite) { $rootPreparationArgs += "--overwrite" }
if (-not $Resume) {
    & $PythonPath @rootPreparationArgs 2>&1 | Write-Host
    if ($LASTEXITCODE -ne 0) { throw "L3 Root 准备失败，exit code=$LASTEXITCODE" }
}
$requiredInputs = @($manifestPath)
if ($Stage -ne "cache") {
    $requiredInputs += @($modelConfigPath, $l1ConfigPath, $l2ConfigPath, $l1CheckpointPath, $l2CheckpointPath)
}
if ($Stage -in @("all", "calibration")) {
    $requiredInputs += @($decisionProfilePath, $policyPath)
}
if ($Stage -in @("calibration", "audit")) {
    $requiredInputs += @($l1HeadPath, $l2HeadPath)
}
if ($Stage -eq "audit") {
    $requiredInputs += @($calibrationPath)
}
foreach ($relative in $requiredInputs) {
    $target = Join-Path $projectRoot $relative
    if (-not (Test-Path -LiteralPath $target -PathType Leaf)) {
        throw "正式 L3 缺少绑定文件：$relative"
    }
}

function Write-AtomicJson {
    param([string]$Path, [object]$Value)
    $parent = Split-Path -Parent $Path
    New-Item -ItemType Directory -Force -Path $parent | Out-Null
    $temporary = Join-Path $parent ("." + [IO.Path]::GetFileName($Path) + "." + $PID + ".tmp")
    $utf8 = [System.Text.UTF8Encoding]::new($false)
    [IO.File]::WriteAllText($temporary, (($Value | ConvertTo-Json -Depth 30) + [Environment]::NewLine), $utf8)
    Move-Item -LiteralPath $temporary -Destination $Path -Force
}

$state = [ordered]@{
    schema_version = "RADAR-LN3-PIPELINE"
    configuration = [ordered]@{
        device = $Device
        batch_size = $BatchSize
        head_batch_size = $HeadBatchSize
        epochs = $Epochs
        learning_rate = $LearningRate
        weight_decay = $WeightDecay
        package_root = "src"
        root = $rootRelative.Replace('\','/')
        profile_path = $decisionProfilePath
    }
    updated_utc = [DateTime]::UtcNow.ToString("o")
    steps = [ordered]@{}
}

if ($Resume) {
    if (-not (Test-Path -LiteralPath $statePath -PathType Leaf)) {
        throw "请求 Resume，但正式 L3 状态文件不存在：$statePath"
    }
    $loaded = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
    if ([string]$loaded.schema_version -notin @($state.schema_version, "RADAR-LN3-FORMAL-L3-PIPELINE-V4")) {
        throw "正式 L3 state schema 不匹配：$($loaded.schema_version)；请使用 -ResetState 开始新轮。"
    }
    foreach ($key in $state.configuration.Keys) {
        $property = $loaded.configuration.PSObject.Properties[$key]
        if ($null -eq $property -and $key -eq "root") { continue }
        if ($null -eq $property -or [string]$property.Value -ne [string]$state.configuration[$key]) {
            throw "续跑参数不一致：$key；如需新一轮请使用 -ResetState。"
        }
    }
    foreach ($property in $loaded.steps.PSObject.Properties) {
        $state.steps[$property.Name] = $property.Value
    }
} elseif (-not $ResetState -and (Test-Path -LiteralPath $statePath -PathType Leaf)) {
    throw "正式 L3 已有状态；续跑请使用 -Resume，开始新一轮请使用 -ResetState。"
}

function Save-State {
    $state.updated_utc = [DateTime]::UtcNow.ToString("o")
    Write-AtomicJson -Path $statePath -Value $state
}

function New-Args {
    param([string[]]$Values)
    $list = [System.Collections.Generic.List[string]]::new()
    foreach ($value in $Values) { [void]$list.Add($value) }
    return ,$list
}

function Add-OverwriteFlag {
    param([System.Collections.Generic.List[string]]$Arguments)
    if ($Overwrite -or $Resume) { [void]$Arguments.Add("--overwrite") }
}

function Add-ValidationFlag {
    param([System.Collections.Generic.List[string]]$Arguments)
    if ($SkipDataValidation) { [void]$Arguments.Add("--skip-data-validation") }
}

function Test-Outputs {
    param([string[]]$Outputs)
    foreach ($relative in $Outputs) {
        if (-not (Test-Path -LiteralPath (Join-Path $projectRoot $relative) -PathType Leaf)) { return $false }
    }
    return $true
}

function Invoke-PackageStep {
    param(
        [string]$Name,
        [System.Collections.Generic.List[string]]$Arguments,
        [string[]]$Outputs,
        [string[]]$Inputs = @()
    )
    $existing = $state.steps[$Name]
    $inputHashes = Get-PipelineInputHashes -ProjectRoot $projectRoot -Inputs $Inputs
    $savedHashProperty = if ($null -eq $existing) { $null } else { $existing.PSObject.Properties["input_sha256"] }
    $savedHashes = if ($null -eq $savedHashProperty) { $null } else { $savedHashProperty.Value }
    $inputsMatch = Test-PipelineInputHashes -SavedHashes $savedHashes -CurrentHashes $inputHashes
    $trainingMatches = $true
    if ($Name -eq "03-knownness-heads") {
        $savedTraining = if ($null -eq $existing) { $null } else { $existing.PSObject.Properties["knownness_training"] }
        $trainingMatches = $null -ne $savedTraining -and ($savedTraining.Value | ConvertTo-Json -Depth 10 -Compress) -eq $knownnessTrainingJson
    }
    if ($Resume -and $null -ne $existing -and $existing.status -eq "completed" -and $inputsMatch -and $trainingMatches -and (Test-Outputs $Outputs)) {
        Write-Host "[trainL3] 续跑跳过：$Name"
        return
    }
    $logPath = Join-Path $logRoot "$Name.log"
    $commandPath = Join-Path $commandRoot "$Name.json"
    Write-AtomicJson -Path $commandPath -Value ([ordered]@{name=$Name; executable=$PythonPath; arguments=@($Arguments.ToArray()); inputs=$inputHashes; outputs=$Outputs; started_utc=[DateTime]::UtcNow.ToString("o")})
    $state.steps[$Name] = [ordered]@{status="running"; log_path=$logPath; command_path=$commandPath; input_sha256=$inputHashes; outputs=$Outputs; error=""}
    if ($Name -eq "03-knownness-heads") {
        $state.steps[$Name].knownness_training = $knownnessTraining
        Write-Host "[trainL3] knownness 配置：$l3ConfigPath"
        Write-Host "[trainL3] 生效训练参数：$knownnessTrainingJson"
    }
    Save-State
    Write-Host "[trainL3] 开始：$Name"
    Write-Host "[trainL3] 日志：$logPath"
    $oldPythonPath = $env:PYTHONPATH
    $srcRoot = Join-Path $projectRoot "src"
    $env:PYTHONPATH = if ([string]::IsNullOrWhiteSpace($oldPythonPath)) { $srcRoot } else { "$srcRoot$([IO.Path]::PathSeparator)$oldPythonPath" }
    try {
        & $PythonPath @($Arguments.ToArray()) 2>&1 | Tee-Object -FilePath $logPath -Append
        $exitCode = if ($null -eq $LASTEXITCODE) { 0 } else { [int]$LASTEXITCODE }
        if ($exitCode -ne 0) { throw "阶段 $Name 失败，exit code=$exitCode。详见 $logPath" }
        if (-not (Test-Outputs $Outputs)) { throw "阶段 $Name 完成但输出不完整。" }
        $state.steps[$Name].status = "completed"
        $state.steps[$Name].exit_code = 0
        Save-State
        Write-Host "[trainL3] 完成：$Name"
    } catch {
        $state.steps[$Name].status = "failed"
        $state.steps[$Name].error = $_.Exception.Message
        Save-State
        throw
    } finally {
        $env:PYTHONPATH = $oldPythonPath
    }
}

$existingLockPid = 0
if (Test-Path -LiteralPath $lockPath -PathType Leaf) {
    $existingLockText = (Get-Content -LiteralPath $lockPath -Raw -ErrorAction SilentlyContinue).Trim()
    $hasExistingPid = [int]::TryParse($existingLockText, [ref]$existingLockPid)
    if ($hasExistingPid -and $existingLockPid -gt 0 -and (Get-Process -Id $existingLockPid -ErrorAction SilentlyContinue)) {
        throw "已有正式 L3 训练进程运行中：PID=$existingLockPid；请等待其结束。"
    }
    Remove-Item -LiteralPath $lockPath -Force
    Write-Host "[trainL3] 已清理上次异常退出留下的 stale lock：$lockPath"
}
$lockStream = [System.IO.File]::Open($lockPath, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::None)
try {
    if ($ResetState -and (Test-Path -LiteralPath $statePath -PathType Leaf)) {
        $archivePath = Join-Path $pipelineRoot ("state.archive." + [DateTime]::UtcNow.ToString("yyyyMMddTHHmmssZ") + ".json")
        Move-Item -LiteralPath $statePath -Destination $archivePath
        Write-Host "[trainL3] 已归档旧状态：$archivePath"
    }
    Save-State
    if ($Stage -in @("all", "cache")) {
        $args = New-Args @("-B","src/scripts/build_l3_cache.py","--project-root",$projectRoot,"--manifest-path",$manifestPath,"--role","head-fit","--output-root","$cacheRoot/head_fit","--progress-interval",[string]$ProgressInterval)
        Add-OverwriteFlag $args; Add-ValidationFlag $args
        Invoke-PackageStep "01-head-fit-cache" $args @($headFitCache)
        $args = New-Args @("-B","src/scripts/build_l3_cache.py","--project-root",$projectRoot,"--manifest-path",$manifestPath,"--role","validation","--output-root","$cacheRoot/validation","--progress-interval",[string]$ProgressInterval)
        Add-OverwriteFlag $args; Add-ValidationFlag $args
        Invoke-PackageStep "02-validation-cache" $args @($validationCache)
    }
    if ($Stage -in @("all", "heads")) {
        $args = New-Args @(
            "-B","src/scripts/train_l3_heads.py","--project-root",$projectRoot,
            "--manifest-path",$manifestPath,"--model-config-path",$modelConfigPath,
            "--l1-checkpoint-path",$l1CheckpointPath,"--l2-checkpoint-path",$l2CheckpointPath,
            "--l1-config-path",$l1ConfigPath,"--l2-config-path",$l2ConfigPath,
            "--fit-feature-cache-manifest-path",$headFitCache,"--validation-feature-cache-manifest-path",$validationCache,
            "--task","both","--device",$Device,"--batch-size",[string]$BatchSize,"--head-batch-size",[string]$HeadBatchSize,
            "--epochs",[string]$Epochs,"--learning-rate",[string]$LearningRate,"--weight-decay",[string]$WeightDecay,
            "--output-root",$headRoot,"--representation-root",$representationRoot,
            "--l3-config-path",$l3ConfigPath
        )
        Add-OverwriteFlag $args; Add-ValidationFlag $args
        if ($SkipDataValidation) { [void]$args.Add("--skip-event-cache-validation") }
        Invoke-PackageStep "03-knownness-heads" $args @($l1HeadPath,$l2HeadPath,"$representationRoot/l1-train.pt","$representationRoot/l1-validation.pt","$representationRoot/l2-train.pt","$representationRoot/l2-validation.pt","$headRoot/l1-knownness-summary.json","$headRoot/l2-knownness-summary.json") -Inputs @($l1CheckpointPath,$l2CheckpointPath,$l1ConfigPath,$l2ConfigPath,$modelConfigPath,$l3ConfigPath,$headFitCache,$validationCache)
    }
    if ($Stage -in @("all", "calibration")) {
        $args = New-Args @("-B","src/scripts/calibrate_l3.py","--project-root",$projectRoot,"--manifest-path",$manifestPath,"--model-config-path",$modelConfigPath,"--l1-checkpoint-path",$l1CheckpointPath,"--l2-checkpoint-path",$l2CheckpointPath,"--l1-config-path",$l1ConfigPath,"--l2-config-path",$l2ConfigPath,"--feature-cache-manifest-path",$validationCache,"--policy-json",$policyPath,"--decision-profile-json",$decisionProfilePath,"--l1-head-path",$l1HeadPath,"--l2-head-path",$l2HeadPath,"--representation-root",$representationRoot,"--device",$Device,"--batch-size",[string]$BatchSize,"--output-path",$calibrationPath)
        Add-OverwriteFlag $args; Add-ValidationFlag $args
        Invoke-PackageStep "04-validation-calibration" $args @($calibrationPath) -Inputs @($l1CheckpointPath,$l2CheckpointPath,$l1ConfigPath,$l2ConfigPath,$modelConfigPath,$validationCache,$policyPath,$decisionProfilePath,$l1HeadPath,$l2HeadPath)
    }
    if ($Stage -in @("all", "audit")) {
        $args = New-Args @("-B","src/scripts/freeze_l3_audit_manifest.py","--project-root",$projectRoot,"--source-manifest-path",$manifestPath,"--calibration-path",$calibrationPath,"--output-manifest-path",$auditManifest)
        Add-OverwriteFlag $args; Add-ValidationFlag $args; Invoke-PackageStep "05-freeze-audit-manifest" $args @($auditManifest) -Inputs @($manifestPath,$calibrationPath)
        $args = New-Args @("-B","src/scripts/build_l3_cache.py","--project-root",$projectRoot,"--manifest-path",$auditManifest,"--role","audit","--output-root","$cacheRoot/audit","--progress-interval",[string]$ProgressInterval)
        Add-OverwriteFlag $args; Add-ValidationFlag $args; Invoke-PackageStep "06-audit-feature-cache" $args @($auditCache) -Inputs @($auditManifest)
        $args = New-Args @("-B","src/scripts/audit_l3.py","--project-root",$projectRoot,"--model-config-path",$modelConfigPath,"--calibration-path",$calibrationPath,"--l1-checkpoint-path",$l1CheckpointPath,"--l2-checkpoint-path",$l2CheckpointPath,"--l1-config-path",$l1ConfigPath,"--l2-config-path",$l2ConfigPath,"--audit-manifest-path",$auditManifest,"--audit-feature-cache-manifest-path",$auditCache,"--audit-split-name","test","--device",$Device,"--batch-size",[string]$BatchSize,"--l1-head-path",$l1HeadPath,"--l2-head-path",$l2HeadPath,"--representation-root",$representationRoot,"--product-manifest-path",$manifestPath,"--thresholds-frozen-before-access","--formal-available","--output-path",$auditedCalibrationPath)
        Add-OverwriteFlag $args; Add-ValidationFlag $args; Invoke-PackageStep "07-full-model-audit" $args @($auditedCalibrationPath) -Inputs @($calibrationPath,$auditManifest,$auditCache,$l1CheckpointPath,$l2CheckpointPath,$l1ConfigPath,$l2ConfigPath,$modelConfigPath,$l1HeadPath,$l2HeadPath)
        $args = New-Args @("-B","src/scripts/report_l3_audit.py","--project-root",$projectRoot,"--model-config-path",$modelConfigPath,"--calibration-path",$auditedCalibrationPath,"--l1-checkpoint-path",$l1CheckpointPath,"--l2-checkpoint-path",$l2CheckpointPath,"--l1-config-path",$l1ConfigPath,"--l2-config-path",$l2ConfigPath,"--audit-manifest-path",$auditManifest,"--audit-feature-cache-manifest-path",$auditCache,"--audit-split-name","test","--device",$Device,"--batch-size",[string]$BatchSize,"--l1-head-path",$l1HeadPath,"--l2-head-path",$l2HeadPath,"--representation-root",$representationRoot,"--output-path",$auditReportPath)
        Add-OverwriteFlag $args; Add-ValidationFlag $args; Invoke-PackageStep "08-audit-report" $args @($auditReportPath) -Inputs @($auditedCalibrationPath,$auditManifest,$auditCache,$l1CheckpointPath,$l2CheckpointPath,$l1ConfigPath,$l2ConfigPath,$modelConfigPath,$l1HeadPath,$l2HeadPath)
    }
    if ($Stage -in @("all", "audit")) {
        $l1FormalEndpoint = Join-Path $projectRoot $l1CheckpointPath
        $l2FormalEndpoint = Join-Path $projectRoot $l2CheckpointPath

        $bindingPath = Join-Path $projectRoot $l3ConfigPath
        $auditedCalibration = Get-Content -LiteralPath (Join-Path $projectRoot $auditedCalibrationPath) -Raw -Encoding UTF8 | ConvertFrom-Json
        $l1CheckpointHash = (Get-FileHash -LiteralPath $l1FormalEndpoint -Algorithm SHA256).Hash.ToLowerInvariant()
        $l2CheckpointHash = (Get-FileHash -LiteralPath $l2FormalEndpoint -Algorithm SHA256).Hash.ToLowerInvariant()
        $l1HeadHash = (Get-FileHash -LiteralPath (Join-Path $projectRoot $l1HeadPath) -Algorithm SHA256).Hash.ToLowerInvariant()
        $l2HeadHash = (Get-FileHash -LiteralPath (Join-Path $projectRoot $l2HeadPath) -Algorithm SHA256).Hash.ToLowerInvariant()
        $calibrationHash = (Get-FileHash -LiteralPath (Join-Path $projectRoot $auditedCalibrationPath) -Algorithm SHA256).Hash.ToLowerInvariant()
        $modelConfigHash = (Get-FileHash -LiteralPath (Join-Path $projectRoot $modelConfigPath) -Algorithm SHA256).Hash.ToLowerInvariant()
        $binding = [ordered]@{
            schema_version = "RADAR-LN3-L3-CONFIG"
            status = [string]$auditedCalibration.status
            research_only = -not [bool]$auditedCalibration.deployment_eligible
            deployment_eligible = [bool]$auditedCalibration.deployment_eligible
            model_config_path = $modelConfigPath
            l1_checkpoint_path = $l1CheckpointPath
            l2_checkpoint_path = $l2CheckpointPath
            l1_knownness_head_path = $l1HeadPath
            l2_knownness_head_path = $l2HeadPath
            l3_calibration_path = $auditedCalibrationPath
            representation_root = $representationRoot
            knownness_source = "independent_l3_head"
            l1_checkpoint_sha256 = $l1CheckpointHash
            l2_checkpoint_sha256 = $l2CheckpointHash
            l1_knownness_head_sha256 = $l1HeadHash
            l2_knownness_head_sha256 = $l2HeadHash
            l3_calibration_sha256 = $calibrationHash
            model_config_sha256 = $modelConfigHash
            l1_endpoint_config_path = $l1ConfigPath
            l2_endpoint_config_path = $l2ConfigPath
            l1_endpoint_config_sha256 = (Get-FileHash -LiteralPath (Join-Path $projectRoot $l1ConfigPath) -Algorithm SHA256).Hash.ToLowerInvariant()
            l2_endpoint_config_sha256 = (Get-FileHash -LiteralPath (Join-Path $projectRoot $l2ConfigPath) -Algorithm SHA256).Hash.ToLowerInvariant()
            policy_path = $policyPath
            policy_sha256 = (Get-FileHash -LiteralPath (Join-Path $projectRoot $policyPath) -Algorithm SHA256).Hash.ToLowerInvariant()
            decision_profile_path = $decisionProfilePath
            decision_profile_sha256 = (Get-FileHash -LiteralPath (Join-Path $projectRoot $decisionProfilePath) -Algorithm SHA256).Hash.ToLowerInvariant()
            training_config_path = $l3ConfigPath
            knownness_training = $knownnessTraining
        }
        Write-AtomicJson -Path $bindingPath -Value $binding
        $state.steps["09-sync-formal-binding"] = [ordered]@{
            status = "completed"
            outputs = @($l3ConfigPath)
            updated_utc = [DateTime]::UtcNow.ToString("o")
        }
        Save-State
        Write-Host "[trainL3] 已刷新正式 L3 配置：$l3ConfigPath；endpoint 直接绑定正式 L1/L2"
    }
    Write-Host "[trainL3] 流程完成；运行状态和日志：$pipelineRoot"
} finally {
    if ($null -ne $lockStream) { $lockStream.Dispose() }
    Remove-Item -LiteralPath $lockPath -Force -ErrorAction SilentlyContinue
}

