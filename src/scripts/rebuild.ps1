[CmdletBinding()]
param(
    [ValidateSet("cuda", "cpu")]
    [string]$Device = "cuda",
    [ValidateRange(1, 64)]
    [int]$BatchSize = 4,
    [ValidateRange(1, 10000)]
    [int]$ProgressInterval = 25,
    [switch]$Resume,
    [string]$PythonPath = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$scriptDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = (Resolve-Path (Join-Path $scriptDirectory "..\..")).Path
$tmpRoot = Join-Path $projectRoot "tmp\rebuild"
$logRoot = Join-Path $tmpRoot "logs"
$commandRoot = Join-Path $tmpRoot "commands"
$statePath = Join-Path $tmpRoot "state.json"
$lockPath = Join-Path $tmpRoot "run.lock"

if ([string]::IsNullOrWhiteSpace($PythonPath)) {
    $PythonPath = Join-Path $projectRoot ".venv\Scripts\python.exe"
} else {
    $PythonPath = (Resolve-Path -LiteralPath $PythonPath).Path
}
if (-not (Test-Path -LiteralPath $PythonPath -PathType Leaf)) {
    throw "Python 解释器不存在：$PythonPath"
}

New-Item -ItemType Directory -Force -Path $tmpRoot, $logRoot, $commandRoot | Out-Null
if (-not $Resume) {
    foreach ($generated in @($logRoot, $commandRoot)) {
        if ([IO.Directory]::Exists($generated)) {
            [IO.Directory]::Delete($generated, $true)
        }
        [IO.Directory]::CreateDirectory($generated) | Out-Null
    }
    if ([IO.File]::Exists($statePath)) {
        [IO.File]::Delete($statePath)
    }
}
if (Test-Path -LiteralPath $lockPath -PathType Leaf) {
    $lockPid = 0
    [int]::TryParse((Get-Content -LiteralPath $lockPath -Raw).Trim(), [ref]$lockPid) | Out-Null
    if ($lockPid -gt 0 -and (Get-Process -Id $lockPid -ErrorAction SilentlyContinue)) {
        throw "已有 rebuild 进程运行中：PID=$lockPid"
    }
    Remove-Item -LiteralPath $lockPath -Force
}
$lockStream = [IO.File]::Open($lockPath, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::None)
$lockBytes = [Text.Encoding]::UTF8.GetBytes([string]$PID)
$lockStream.Write($lockBytes, 0, $lockBytes.Length)
$lockStream.Flush()

$utf8 = [Text.UTF8Encoding]::new($false)
$state = [ordered]@{
    schema_version = "RADAR-LN3-REBUILD-PIPELINE-V1"
    status = "running"
    updated_utc = [DateTime]::UtcNow.ToString("o")
    configuration = [ordered]@{
        device = $Device
        batch_size = $BatchSize
        progress_interval = $ProgressInterval
    }
    steps = [ordered]@{}
}
if ($Resume) {
    if (-not [IO.File]::Exists($statePath)) {
        throw "请求 Resume，但 rebuild state.json 不存在：$statePath"
    }
    $loaded = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
    if ([string]$loaded.schema_version -ne "RADAR-LN3-REBUILD-PIPELINE-V1") {
        throw "rebuild state schema 不受支持：$($loaded.schema_version)"
    }
    foreach ($key in $state.configuration.Keys) {
        $property = $loaded.configuration.PSObject.Properties[$key]
        if ($null -eq $property -or [string]$property.Value -ne [string]$state.configuration[$key]) {
            throw "rebuild 续跑参数不一致：$key；请使用不带 -Resume 的新一轮。"
        }
    }
    if ($null -ne $loaded.steps) {
        foreach ($property in $loaded.steps.PSObject.Properties) {
            $state.steps[$property.Name] = $property.Value
        }
    }
}

function Write-State {
    $state.updated_utc = [DateTime]::UtcNow.ToString("o")
    $temporary = Join-Path $tmpRoot (".state." + $PID + ".tmp")
    [IO.File]::WriteAllText($temporary, (($state | ConvertTo-Json -Depth 20) + [Environment]::NewLine), $utf8)
    Move-Item -LiteralPath $temporary -Destination $statePath -Force
}

function Set-Step {
    param([string]$Name, [string]$Status, [string]$LogPath = "", [string]$CommandPath = "", [string]$ErrorMessage = "")
    $state.steps[$Name] = [ordered]@{
        status = $Status
        updated_utc = [DateTime]::UtcNow.ToString("o")
        log_path = $LogPath
        command_path = $CommandPath
        error = $ErrorMessage
    }
    Write-State
}

function Invoke-Step {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$Executable,
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [Parameter(Mandatory = $true)][string[]]$Outputs
    )
    $old = $state.steps[$Name]
    if ($Resume -and $null -ne $old -and $old.status -eq "completed" -and ($Outputs | ForEach-Object { Test-Path -LiteralPath (Join-Path $projectRoot $_) -PathType Leaf } | Where-Object { -not $_ }).Count -eq 0) {
        Write-Host "[rebuild] 续跑跳过：$Name"
        return
    }
    $index = $state.steps.Count + 1
    $logPath = Join-Path $logRoot ("{0:D2}-{1}.log" -f $index, ($Name -replace "[^A-Za-z0-9_.-]", "_"))
    $commandPath = Join-Path $commandRoot ("{0:D2}-{1}.json" -f $index, ($Name -replace "[^A-Za-z0-9_.-]", "_"))
    New-Item -ItemType Directory -Force -Path $tmpRoot, $logRoot, $commandRoot, (Split-Path -Parent $logPath), (Split-Path -Parent $commandPath) | Out-Null
    $command = [ordered]@{ schema_version = "RADAR-LN3-REBUILD-COMMAND-V1"; name = $Name; executable = $Executable; arguments = @($Arguments); outputs = @($Outputs); started_utc = [DateTime]::UtcNow.ToString("o") }
    [IO.File]::WriteAllText($commandPath, (($command | ConvertTo-Json -Depth 10) + [Environment]::NewLine), $utf8)
    Set-Step -Name $Name -Status "running" -LogPath $logPath -CommandPath $commandPath
    Write-Host "[rebuild] 开始：$Name"
    Write-Host "[rebuild] 日志：$logPath"
    $oldPythonPath = $env:PYTHONPATH
    $env:PYTHONPATH = Join-Path $projectRoot "src"
    try {
        & $Executable @Arguments 2>&1 | Tee-Object -FilePath $logPath -Append
        $exitCode = if ($null -eq $LASTEXITCODE) { 0 } else { [int]$LASTEXITCODE }
        if ($exitCode -ne 0) { throw "阶段 $Name 失败，exit code=$exitCode" }
        foreach ($relative in $Outputs) {
            if (-not (Test-Path -LiteralPath (Join-Path $projectRoot $relative) -PathType Leaf)) { throw "阶段 $Name 输出缺失：$relative" }
        }
        Set-Step -Name $Name -Status "completed" -LogPath $logPath -CommandPath $commandPath
        Write-Host "[rebuild] 完成：$Name"
    } catch {
        Set-Step -Name $Name -Status "failed" -LogPath $logPath -CommandPath $commandPath -ErrorMessage $_.Exception.Message
        throw
    } finally {
        $env:PYTHONPATH = $oldPythonPath
    }
}

try {
    Write-State
    Write-Host "[rebuild] 项目根目录：$projectRoot"
    Write-Host "[rebuild] 日志目录：$tmpRoot"

    if (-not $Resume) {
        Write-Host "[rebuild] 清理旧 derived cache、anchor 和实验状态"
        foreach ($relative in @("data/cq08/cache", "data/cq08/rd/canonical", "tmp/L2", "tmp/L3")) {
            $target = Join-Path $projectRoot $relative
            if (Test-Path -LiteralPath $target) { Remove-Item -LiteralPath $target -Recurse -Force }
        }
        New-Item -ItemType Directory -Force -Path (Join-Path $projectRoot "tmp/L2"), (Join-Path $projectRoot "tmp/L3") | Out-Null
    }

    Invoke-Step -Name "data_index" -Executable $PythonPath -Arguments @(
        "-B", "-m", "core.cli", "--handle-raw-data", "--project-root", $projectRoot,
        "--manifest-path", "src/config/manifests.json", "--cache-path", "data/cq08/cache/event-source-cache.jsonl", "--overwrite"
    ) -Outputs @("src/config/manifests.json", "data/cq08/cache/event-source-cache.jsonl", "data/cq08/tr/cq08-tr-event-index.jsonl", "data/cq08/rd/cq08-rd-frame-index.jsonl")

    Invoke-Step -Name "canonical_rd" -Executable $PythonPath -Arguments @(
        "-B", "src/scripts/materialize_canonical_rd.py", "--project-root", $projectRoot,
        "--manifest-path", "src/config/manifests.json", "--source-frame-index-path", "data/processed/cq08_rd_physical_canonical_frame_index_v3_1.csv",
        "--output-root", "data/cq08/rd/canonical", "--overwrite", "--skip-data-validation"
    ) -Outputs @("data/cq08/rd/canonical/canonical-rd-manifest.json", "data/cq08/rd/canonical/canonical-rd-event-index.jsonl")

    $trainScript = Join-Path $projectRoot "src/scripts/train.ps1"
    $powerShellHost = (Get-Command pwsh -ErrorAction SilentlyContinue).Source
    if ([string]::IsNullOrWhiteSpace($powerShellHost)) {
        $powerShellHost = (Get-Command powershell -ErrorAction Stop).Source
    }
    $psArgs = @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $trainScript, "-Stage", "cache", "-L2Mode", "residual", "-Device", $Device, "-BatchSize", [string]$BatchSize, "-ProgressInterval", [string]$ProgressInterval, "-RebuildCache", "-SkipDataValidation")
    Invoke-Step -Name "feature_cache" -Executable $powerShellHost -Arguments $psArgs -Outputs @("data/cq08/cache/features/feature-cache-manifest.json", "data/cq08/cache/features/l1/train/feature-cache-manifest.json", "data/cq08/cache/features/l1/validation/feature-cache-manifest.json", "data/cq08/cache/features/l2/train/feature-cache-manifest.json", "data/cq08/cache/features/l2/validation/feature-cache-manifest.json")

    Invoke-Step -Name "baseline_anchor" -Executable $PythonPath -Arguments @(
        "-B", "src/scripts/ensure_l2_baseline_anchor.py", "--project-root", $projectRoot,
        "--manifest-path", "src/config/manifests.json", "--feature-cache-manifest-path", "data/cq08/cache/features/feature-cache-manifest.json",
        "--output-path", "data/cq08/cache/l2-baseline-anchor.pt", "--device", $Device, "--batch-size", [string]$BatchSize, "--rebuild", "--skip-data-validation"
    ) -Outputs @("data/cq08/cache/l2-baseline-anchor.pt")

    $state.status = "completed"
    Write-State
    Write-Host "[rebuild] 全部完成；后续可运行 train.ps1、trainL2.ps1 和 trainL3.ps1。"
} catch {
    $state.status = "failed"
    $state.error = $_.Exception.Message
    Write-State
    throw
} finally {
    if ($null -ne $lockStream) { $lockStream.Dispose() }
    Remove-Item -LiteralPath $lockPath -Force -ErrorAction SilentlyContinue
}
