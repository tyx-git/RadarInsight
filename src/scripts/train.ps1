[CmdletBinding()]
param(
    [ValidateSet("cuda", "cpu")]
    [string]$Device = "cuda",

    [ValidateRange(1, 64)]
    [int]$BatchSize = 4,

    [ValidateRange(1, 10000)]
    [int]$ProgressInterval = 25,

    [switch]$RefreshData,

    [switch]$StopAfterL1,

    [switch]$ResetState,

    [string]$PythonPath = ""
    ,[string]$Root = "artifacts/radar_ln3"
    ,[switch]$Overwrite
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$scriptDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = (Resolve-Path (Join-Path $scriptDirectory "..\..")).Path
. (Join-Path $scriptDirectory "common\pipeline_inputs.ps1")
$rootPath = if ([IO.Path]::IsPathRooted($Root)) { [IO.Path]::GetFullPath($Root) } else { [IO.Path]::GetFullPath((Join-Path $projectRoot $Root)) }
$rootRelative = [IO.Path]::GetRelativePath($projectRoot, $rootPath)
if ($rootRelative -eq ".." -or $rootRelative.StartsWith("..$([IO.Path]::DirectorySeparatorChar)") -or [IO.Path]::IsPathRooted($rootRelative)) { throw "Root 必须位于项目目录内：$Root" }
$pipelineRoot = Join-Path $rootPath "pipeline"
$l1FinalRel = ($rootRelative.Replace("\","/") + "/l1/l1.pt")
$l1SummaryRel = ($rootRelative.Replace("\","/") + "/l1/training-summary.json")
$l2FinalRel = ($rootRelative.Replace("\","/") + "/l2/l2.pt")
$l2SummaryRel = ($rootRelative.Replace("\","/") + "/l2/training-summary.json")
$l3CalibrationRel = ($rootRelative.Replace("\","/") + "/l3/radar_ln3_calibration.json")
$l3ReportRel = ($rootRelative.Replace("\","/") + "/l3/radar_ln3_audit_report.json")
$logRoot = Join-Path $pipelineRoot "logs"
$commandRoot = Join-Path $pipelineRoot "commands"
$statePath = Join-Path $pipelineRoot "state.json"
$trainScript = Join-Path $projectRoot "src\scripts\train_endpoint.ps1"
$powerShellHost = (Get-Command pwsh.exe -ErrorAction SilentlyContinue).Source
if ([string]::IsNullOrWhiteSpace($powerShellHost)) {
    $powerShellHost = (Get-Command powershell.exe -ErrorAction SilentlyContinue).Source
}
if ([string]::IsNullOrWhiteSpace($powerShellHost)) {
    throw "找不到 PowerShell 主机，无法调用 train.ps1。"
}

if ([string]::IsNullOrWhiteSpace($PythonPath)) {
    $PythonPath = Join-Path $projectRoot ".venv\Scripts\python.exe"
} else {
    $PythonPath = (Resolve-Path -LiteralPath $PythonPath).Path
}
if (-not (Test-Path -LiteralPath $PythonPath -PathType Leaf)) {
    throw "Python 解释器不存在：$PythonPath"
}
if (-not (Test-Path -LiteralPath $trainScript -PathType Leaf)) {
    throw "训练脚本不存在：$trainScript"
}

New-Item -ItemType Directory -Force -Path $pipelineRoot, $logRoot, $commandRoot | Out-Null

function Write-AtomicText {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Text
    )

    $parent = Split-Path -Parent $Path
    New-Item -ItemType Directory -Force -Path $parent | Out-Null
    $temporary = Join-Path $parent ("." + [IO.Path]::GetFileName($Path) + "." + $PID + ".tmp")
    $utf8 = New-Object System.Text.UTF8Encoding($false)
    [IO.File]::WriteAllText($temporary, $Text, $utf8)
    Move-Item -LiteralPath $temporary -Destination $Path -Force
}

function Save-PipelineState {
    $script:State.updated_utc = [DateTime]::UtcNow.ToString("o")
    $payload = $script:State | ConvertTo-Json -Depth 30
    Write-AtomicText -Path $statePath -Text ($payload + [Environment]::NewLine)
}

function Get-StepState {
    param([Parameter(Mandatory = $true)][string]$Name)

    $property = $script:State.steps.PSObject.Properties[$Name]
    if ($null -eq $property) {
        return $null
    }
    return $property.Value
}

function Set-StepState {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$Status,
        [int]$ExitCode = 0,
        [string]$LogPath = "",
        [string]$CommandPath = "",
        [string[]]$Outputs = @(),
        [string]$ErrorMessage = "",
        [System.Collections.IDictionary]$InputHashes = [ordered]@{}
    )

    $old = Get-StepState -Name $Name
    $started = if ($null -ne $old -and $old.started_utc) {
        [string]$old.started_utc
    } else {
        [DateTime]::UtcNow.ToString("o")
    }
    $completed = $null
    if ($Status -eq "completed" -or $Status -eq "skipped") {
        $completed = [DateTime]::UtcNow.ToString("o")
    }
    $value = [pscustomobject]@{
        status = $Status
        started_utc = $started
        updated_utc = [DateTime]::UtcNow.ToString("o")
        completed_utc = $completed
        exit_code = [int]$ExitCode
        log_path = $LogPath
        command_path = $CommandPath
        input_sha256 = $InputHashes
        outputs = @($Outputs)
        error = $ErrorMessage
    }
    $property = $script:State.steps.PSObject.Properties[$Name]
    if ($null -eq $property) {
        $script:State.steps | Add-Member -MemberType NoteProperty -Name $Name -Value $value
    } else {
        $property.Value = $value
    }
    Save-PipelineState
}

function Test-FreshOutputs {
    param([Parameter(Mandatory = $true)][string[]]$Outputs)

    $runStart = [DateTime]::Parse([string]$script:State.run_started_utc).ToUniversalTime()
    return Test-PipelineOutputs -ProjectRoot $projectRoot -Outputs $Outputs -RunStartedUtc $runStart
}

function Test-ExistingOutputs {
    param([Parameter(Mandatory = $true)][string[]]$Outputs)

    return Test-PipelineOutputs -ProjectRoot $projectRoot -Outputs $Outputs -RunStartedUtc ([DateTime]::MinValue) -AllowExisting
}

function Invoke-ResumableStep {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$Executable,
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [Parameter(Mandatory = $true)][string[]]$Outputs,
        [string[]]$Inputs = @(),
        [switch]$AllowExistingOutputs
    )

    $existing = Get-StepState -Name $Name
    $inputHashes = Get-PipelineInputHashes -ProjectRoot $projectRoot -Inputs $Inputs
    $savedHashProperty = if ($null -eq $existing) { $null } else { $existing.PSObject.Properties["input_sha256"] }
    $savedHashes = if ($null -eq $savedHashProperty) { $null } else { $savedHashProperty.Value }
    $inputsMatch = Test-PipelineInputHashes -SavedHashes $savedHashes -CurrentHashes $inputHashes
    $outputsReady = if ($AllowExistingOutputs) { Test-ExistingOutputs -Outputs $Outputs } else { Test-FreshOutputs -Outputs $Outputs }
    if ($null -ne $existing -and [string]$existing.status -eq "completed" -and $inputsMatch -and $outputsReady) {
        Write-Host "[pipeline] 跳过已完成阶段：$Name"
        return
    }

    $index = 0
    foreach ($property in $script:State.steps.PSObject.Properties) {
        $index += 1
    }
    $safeName = $Name -replace "[^A-Za-z0-9_.-]", "_"
    $logPath = Join-Path $logRoot (("{0:D2}-{1}.log" -f ($index + 1), $safeName))
    $commandPath = Join-Path $commandRoot (("{0:D2}-{1}.json" -f ($index + 1), $safeName))
    # 阶段之间可能发生人工清理或外部归档；每次执行前重新确保日志和命令目录存在。
    New-Item -ItemType Directory -Force -Path $pipelineRoot, $logRoot, $commandRoot, (Split-Path -Parent $logPath), (Split-Path -Parent $commandPath) | Out-Null
    $commandPayload = [ordered]@{
        schema_version = "RADAR-LN3-PIPELINE-COMMAND-V1"
        name = $Name
        executable = $Executable
        arguments = @($Arguments)
        input_sha256 = $inputHashes
        started_utc = [DateTime]::UtcNow.ToString("o")
        outputs = @($Outputs)
    }
    Write-AtomicText -Path $commandPath -Text (($commandPayload | ConvertTo-Json -Depth 10) + [Environment]::NewLine)
    Set-StepState -Name $Name -Status "running" -ExitCode 0 -LogPath $logPath -CommandPath $commandPath -Outputs $Outputs -InputHashes $inputHashes
    Write-Host "[pipeline] 开始阶段：$Name"
    Write-Host "[pipeline] 日志：$logPath"

    $oldLastExitCode = $null
    $lastExitVariable = Get-Variable -Name LASTEXITCODE -Scope Global -ErrorAction SilentlyContinue
    if ($null -ne $lastExitVariable) {
        $oldLastExitCode = $lastExitVariable.Value
    }
    $global:LASTEXITCODE = 0
    try {
        # 数组展开对 .ps1 只产生位置参数；通过 -File 调用才能保留命名参数。
        $invokeExecutable = $Executable
        $invokeArguments = @($Arguments)
        if ([IO.Path]::GetExtension($Executable) -ieq ".ps1") {
            $invokeExecutable = $powerShellHost
            $invokeArguments = @(
                "-NoProfile",
                "-ExecutionPolicy", "Bypass",
                "-File", $Executable
            ) + @($Arguments)
        }
        & $invokeExecutable @invokeArguments 2>&1 | Tee-Object -FilePath $logPath -Append
        $exitCode = if ($null -eq $LASTEXITCODE) { 0 } else { [int]$LASTEXITCODE }
        if ($exitCode -ne 0) {
            throw "阶段 $Name 失败，exit code=$exitCode。详见 $logPath"
        }
        $outputsReady = if ($AllowExistingOutputs) { Test-ExistingOutputs -Outputs $Outputs } else { Test-FreshOutputs -Outputs $Outputs }
        if (-not $outputsReady) {
            throw "阶段 $Name 返回成功，但输出未完整落盘或不是本轮生成：$($Outputs -join ', ')"
        }
        Set-StepState -Name $Name -Status "completed" -ExitCode 0 -LogPath $logPath -CommandPath $commandPath -Outputs $Outputs -InputHashes $inputHashes
        Write-Host "[pipeline] 完成阶段：$Name"
    } catch {
        $exitCode = if ($null -eq $LASTEXITCODE) { 1 } else { [int]$LASTEXITCODE }
        Set-StepState -Name $Name -Status "failed" -ExitCode $exitCode -LogPath $logPath -CommandPath $commandPath -Outputs $Outputs -ErrorMessage $_.Exception.Message -InputHashes $inputHashes
        throw
    } finally {
        if ($null -eq $oldLastExitCode) {
            Remove-Variable -Name LASTEXITCODE -Scope Global -ErrorAction SilentlyContinue
        } else {
            $global:LASTEXITCODE = $oldLastExitCode
        }
    }
}

function Mark-ReusedStep {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string[]]$Outputs
    )
    if (-not (Test-ExistingOutputs -Outputs $Outputs)) {
        throw "阶段 $Name 选择复用，但所需产物不存在：$($Outputs -join ', ')"
    }
    Set-StepState -Name $Name -Status "skipped" -ExitCode 0 -Outputs $Outputs
    Write-Host "[pipeline] 复用现有数据阶段：$Name"
}

$l3KnownnessConfigPath = Join-Path $projectRoot "src\config\radar_ln3_l3.json"
$l3KnownnessConfig = Get-Content -LiteralPath $l3KnownnessConfigPath -Raw -Encoding UTF8 | ConvertFrom-Json
$l3KnownnessTrainingJson = $l3KnownnessConfig.knownness_training | ConvertTo-Json -Depth 10 -Compress

$expectedConfig = [ordered]@{
    schema_version = "RADAR-LN3-PIPELINE-V1"
    device = $Device
    batch_size = $BatchSize
    progress_interval = $ProgressInterval
    refresh_data = [bool]$RefreshData
    stop_after_l1 = [bool]$StopAfterL1
    l2_mode = "residual"
    max_sequence_length = 29
    l3_representation_schema = "RADAR-LN3-ENDPOINT-REPRESENTATION-V3"
    l3_head_schema = "RADAR-LN3-KNOWNNESS-HEAD-V2"
    l3_knownness_training = $l3KnownnessTrainingJson
    l3_policy_path = "src/config/radar_ln3_policy.json"
    l3_representation_root = "artifacts/radar_ln3/l3/representations"
}

if ($ResetState -and (Test-Path -LiteralPath $statePath -PathType Leaf)) {
    $archive = Join-Path $pipelineRoot ("state.archive." + [DateTime]::UtcNow.ToString("yyyyMMddTHHmmssZ") + ".json")
    Move-Item -LiteralPath $statePath -Destination $archive -Force
    Write-Host "[pipeline] 已归档旧状态：$archive"
}

if (Test-Path -LiteralPath $statePath -PathType Leaf) {
    $script:State = Get-Content -LiteralPath $statePath -Raw -Encoding UTF8 | ConvertFrom-Json
    if ([string]$script:State.schema_version -ne "RADAR-LN3-PIPELINE-V1") {
        throw "pipeline state schema_version 不受支持：$($script:State.schema_version)"
    }
    foreach ($key in $expectedConfig.Keys) {
        $configProperty = $script:State.config.PSObject.Properties[$key]
        if ($null -eq $configProperty) {
            $script:State.config | Add-Member -MemberType NoteProperty -Name $key -Value $expectedConfig[$key]
            continue
        }
        $actual = [string]$configProperty.Value
        $expected = [string]$expectedConfig[$key]
        if (
            $key -eq "refresh_data" -and
            $actual -eq "True" -and
            $expected -eq "False"
        ) {
            # 上一轮已经完成数据刷新；续跑时关闭刷新不会改变已落盘数据。
            continue
        }
        if ($actual -ne $expected) {
            throw "续跑参数与已有状态不一致：$key，已有=$actual，当前=$expected；如需新一轮请加 -ResetState。"
        }
    }
    if ($null -eq $script:State.PSObject.Properties["status"]) {
        $script:State | Add-Member -MemberType NoteProperty -Name status -Value "pending"
    }
    if ($null -eq $script:State.PSObject.Properties["failure"]) {
        $script:State | Add-Member -MemberType NoteProperty -Name failure -Value $null
    }
} else {
    $script:State = [pscustomobject]@{}
    $script:State | Add-Member -MemberType NoteProperty -Name schema_version -Value "RADAR-LN3-PIPELINE-V1"
    $script:State | Add-Member -MemberType NoteProperty -Name run_started_utc -Value ([DateTime]::UtcNow.ToString("o"))
    $script:State | Add-Member -MemberType NoteProperty -Name updated_utc -Value ([DateTime]::UtcNow.ToString("o"))
    $script:State | Add-Member -MemberType NoteProperty -Name config -Value ([pscustomobject]$expectedConfig)
    $script:State | Add-Member -MemberType NoteProperty -Name steps -Value ([pscustomobject]@{})
    $script:State | Add-Member -MemberType NoteProperty -Name status -Value "pending"
    $script:State | Add-Member -MemberType NoteProperty -Name failure -Value $null
    Save-PipelineState
}

$oldPythonPath = $env:PYTHONPATH
$srcRoot = Join-Path $projectRoot "src"
if ([string]::IsNullOrWhiteSpace($oldPythonPath)) {
    $env:PYTHONPATH = $srcRoot
} else {
    $env:PYTHONPATH = "$srcRoot$([IO.Path]::PathSeparator)$oldPythonPath"
}

try {
    if ($RefreshData) {
        Invoke-ResumableStep -Name "extend_split" -Executable $PythonPath -Arguments @(
            "-B", "src/scripts/extend_cq08_split.py",
            "--project-root", $projectRoot,
            "--split-path", "data/manifests/cq08_grouped_split.json",
            "--parent-id", "cq08|track|3338",
            "--parent-id", "cq08|track|634",
            "--parent-id", "cq08|track|785",
            "--split", "train",
            "--reason", "newly discovered CQ08 parents; explicit train-only append",
            "--overwrite"
        ) -Outputs @("data/manifests/cq08_grouped_split.json")
    }
    $dataOutputs = @(
        "src/config/manifests.json",
        "data/cq08/cache/event-source-cache.jsonl",
        "data/cq08/tr/cq08-tr-event-index.jsonl",
        "data/cq08/rd/cq08-rd-frame-index.jsonl"
    )
    if ($RefreshData) {
        Invoke-ResumableStep -Name "data_index" -Executable $PythonPath -Arguments @(
            "-B", "-m", "core.cli", "--handle-raw-data",
            "--project-root", $projectRoot,
            "--manifest-path", "src/config/manifests.json",
            "--cache-path", "data/cq08/cache/event-source-cache.jsonl",
            "--overwrite"
        ) -Outputs $dataOutputs
    } else {
        Mark-ReusedStep -Name "data_index" -Outputs $dataOutputs
    }

    $canonicalOutputs = @(
        "data/cq08/rd/canonical/canonical-rd-manifest.json",
        "data/cq08/rd/canonical/canonical-rd-event-index.jsonl"
    )
    if ($RefreshData) {
        Invoke-ResumableStep -Name "canonical_rd" -Executable $PythonPath -Arguments @(
            "-B", "src/scripts/materialize_canonical_rd.py",
            "--project-root", $projectRoot,
            "--manifest-path", "src/config/manifests.json",
            "--source-frame-index-path", "data/processed/cq08_rd_physical_canonical_frame_index_v3_1.csv",
            "--output-root", "data/cq08/rd/canonical",
            "--overwrite", "--skip-data-validation"
        ) -Outputs $canonicalOutputs
    } else {
        Mark-ReusedStep -Name "canonical_rd" -Outputs $canonicalOutputs
    }

    $featureCacheOutputs = @(
        "data/cq08/cache/features/feature-cache-manifest.json",
        "data/cq08/cache/features/l1/train/feature-cache-manifest.json",
        "data/cq08/cache/features/l1/validation/feature-cache-manifest.json",
        "data/cq08/cache/features/l2/train/feature-cache-manifest.json",
        "data/cq08/cache/features/l2/validation/feature-cache-manifest.json"
    )
    if ($RefreshData) {
        $featureCacheArguments = @(
            "-Stage", "cache", "-L2Mode", "residual", "-Device", $Device,
            "-ProgressInterval", [string]$ProgressInterval, "-BatchSize", [string]$BatchSize,
            "-PythonPath", $PythonPath, "-RebuildCache", "-SkipDataValidation"
        )
        Invoke-ResumableStep -Name "feature_cache" -Executable $trainScript -Arguments $featureCacheArguments -Outputs $featureCacheOutputs
    } else {
        Mark-ReusedStep -Name "feature_cache" -Outputs $featureCacheOutputs
    }

    Invoke-ResumableStep -Name "l1_train" -Executable $trainScript -Arguments @(
        "-Root", $rootPath, "-Stage", "l1", "-L2Mode", "residual", "-Device", $Device,
        "-ModelConfigPath", "src/config/model.json",
        "-ProgressInterval", [string]$ProgressInterval, "-BatchSize", [string]$BatchSize,
        "-PythonPath", $PythonPath, "-ReplaceExisting"
    ) -Outputs @(
        $l1FinalRel,
        $l1SummaryRel
    ) -Inputs @("src/config/model.json")

    if ($StopAfterL1) {
        $script:State.status = "completed"
        Save-PipelineState
        Write-Host "[pipeline] 已完成 L1，按 -StopAfterL1 停止；后续请运行 trainL2.ps1 和 trainL3.ps1。"
        return
    }

    Invoke-ResumableStep -Name "l2_baseline_anchor" -Executable $PythonPath -Arguments @(
        "-B", "src/scripts/ensure_l2_baseline_anchor.py",
        "--project-root", $projectRoot,
        "--manifest-path", "src/config/manifests.json",
        "--feature-cache-manifest-path", "data/cq08/cache/features/feature-cache-manifest.json",
        "--output-path", "data/cq08/cache/l2-baseline-anchor.pt",
        "--device", $Device,
        "--batch-size", [string]$BatchSize,
        "--skip-data-validation"
    ) -Outputs @("data/cq08/cache/l2-baseline-anchor.pt") -Inputs @("src/config/manifests.json") -AllowExistingOutputs

    $l2Arguments = @(
        "-Root", $rootPath, "-Stage", "l2", "-L2Mode", "residual", "-Device", $Device,
        "-ModelConfigPath", "src/config/model.json",
        "-ProgressInterval", [string]$ProgressInterval, "-BatchSize", [string]$BatchSize,
        "-PythonPath", $PythonPath, "-ReplaceExisting"
    )
    $previousL2 = Get-StepState -Name "l2_train"
    $l2EpochRoot = Join-Path $projectRoot "artifacts/radar_ln3/l2/epochs"
    $l2Final = Join-Path $projectRoot $l2FinalRel
    $l2Epochs = @(Get-ChildItem -LiteralPath $l2EpochRoot -Filter "epoch-*.pt" -File -ErrorAction SilentlyContinue)
    if (
        $null -ne $previousL2 -and
        [string]$previousL2.status -in @("running", "failed") -and
        -not (Test-Path -LiteralPath $l2Final -PathType Leaf) -and
        $l2Epochs.Count -gt 0
    ) {
        $l2Arguments = @(
            "-Root", $rootPath, "-Stage", "l2", "-L2Mode", "residual", "-Device", $Device,
            "-ModelConfigPath", "src/config/model.json",
            "-ProgressInterval", [string]$ProgressInterval, "-BatchSize", [string]$BatchSize,
            "-PythonPath", $PythonPath, "-Resume"
        )
        Write-Host "[pipeline] 检测到可恢复的 L2 epoch checkpoint，将使用 -Resume。"
    }
    Invoke-ResumableStep -Name "l2_train" -Executable $trainScript -Arguments $l2Arguments -Outputs @(
        $l2FinalRel,
        $l2SummaryRel
    ) -Inputs @("src/config/model.json")

    $formalL3Script = Join-Path $projectRoot "src\scripts\trainL3.ps1"
    if (-not (Test-Path -LiteralPath $formalL3Script -PathType Leaf)) {
        throw "正式 L3 入口不存在：$formalL3Script"
    }
    $formalL3State = Join-Path $projectRoot "artifacts\radar_ln3\pipeline\l3\state.json"
    $formalL3Arguments = @(
        "-NoProfile", "-NonInteractive", "-File", $formalL3Script,
        "-Root", $rootPath,
        "-Stage", "all", "-Device", $Device,
        "-BatchSize", [string]$BatchSize,
        "-ProgressInterval", [string]$ProgressInterval,
        "-SkipDataValidation"
    )
    if ($ResetState -or -not (Test-Path -LiteralPath $formalL3State -PathType Leaf)) {
        $formalL3Arguments += @("-Overwrite", "-ResetState")
    } else {
        $formalL3Arguments += "-Resume"
    }
    Invoke-ResumableStep -Name "l3_formal" -Executable $powerShellHost -Arguments $formalL3Arguments -Inputs @(
        "src/scripts/trainL3.ps1",
        "src/scripts/common/pipeline_inputs.ps1",
        "src/config/model.json",
        "src/config/model.json",
        "src/config/model.json",
        "src/config/radar_ln3_decision.json",
        "src/config/radar_ln3_policy.json",
        $l1FinalRel,
        $l2FinalRel
    ) -Outputs @(
        $l3CalibrationRel,
        $l3ReportRel,
        $l1FinalRel,
        $l2FinalRel,
        "src/config/radar_ln3_l3.json"
    )

    $script:State.status = "completed"
    Save-PipelineState
    Write-Host "[pipeline] 全链路完成。状态文件：$statePath"
} catch {
    $script:State.status = "failed"
    $script:State.failure = $_.Exception.Message
    Save-PipelineState
    throw
} finally {
    $env:PYTHONPATH = $oldPythonPath
}
