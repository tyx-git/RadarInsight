[CmdletBinding()]
param(
    [ValidateSet('prepare', 'train', 'evaluate', 'all')]
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
$defaultProjectRoot = (Resolve-Path -LiteralPath (Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $srcRoot)))).Path
$projectPath = [System.IO.Path]::GetFullPath($(if ([string]::IsNullOrWhiteSpace($ProjectRoot)) { $defaultProjectRoot } else { $ProjectRoot }))
$python = Join-Path $projectPath '.venv\Scripts\python.exe'
$artifactPath = Join-Path $projectPath 'artifacts\RADAR-LN3-V25a-cq08-liquid-known-gate-course-preclip-gradient-telemetry-seed43-formal'
$readinessPath = Join-Path $projectPath 'src-v3\processed\report\RADAR-LN3-V25a-known-gate-course-preclip-gradient-telemetry-training-readiness.json'
$reportPath = Join-Path $projectPath 'src-v3\processed\report\RADAR-LN3-V25a-known-gate-course-preclip-gradient-telemetry-audit.json'
$minimumFreeGpuMemoryMiB = 4096

function Invoke-RadarPython {
    param([Parameter(Mandatory = $true)][string[]]$Arguments)
    $runtimeArguments = @('-X', 'utf8') + $Arguments
    $display = $python + ' ' + ($runtimeArguments -join ' ')
    Write-Host ('> ' + $display) -ForegroundColor Cyan
    Push-Location $srcRoot
    try {
        & $python @runtimeArguments | Out-Host
        $code = [int]$LASTEXITCODE
    }
    finally { Pop-Location }
    if ($code -ne 0) { throw ('Python command failed with exit code ' + $code + ': ' + $display) }
}

function Assert-File {
    param([Parameter(Mandatory = $true)][string]$Path, [Parameter(Mandatory = $true)][string]$Description)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { throw ($Description + ' is missing: ' + $Path) }
}

function Read-Utf8Json {
    param([Parameter(Mandatory = $true)][string]$Path)
    ConvertFrom-Json -InputObject ([System.IO.File]::ReadAllText($Path, [System.Text.UTF8Encoding]::new($false, $true)))
}

function Assert-Readiness {
    Assert-File $readinessPath 'V25a readiness'
    $readiness = Read-Utf8Json $readinessPath
    $issues = [System.Collections.Generic.List[string]]::new()
    if ([string]$readiness.status -ne 'passed') { $issues.Add('status') }
    if ([string]$readiness.schema_version -ne 'RADAR-LN3-V25a-known-gate-course-preclip-gradient-telemetry-training-readiness-v1') { $issues.Add('schema_version') }
    if ([string]$readiness.variant -ne 'v25a_course_preclip_gradient_telemetry') { $issues.Add('variant') }
    if ([int]$readiness.known_train_parent_count -ne 986 -or [int]$readiness.known_validation_parent_count -ne 232 -or [int]$readiness.validation_clutter_parent_support -ne 9) { $issues.Add('support') }
    if ([int]$readiness.unknown_training_exposure -ne 0) { $issues.Add('unknown_training_exposure') }
    if ([bool]$readiness.validation_used_for_readiness_gate) { $issues.Add('validation_used_for_readiness_gate') }
    if ([int]$readiness.test_payload_access_count -ne 0 -or [bool]$readiness.deployment_eligible) { $issues.Add('sealed_scope') }
    if ([string]$readiness.formal_training -ne 'not_executed' -and -not (Test-Path -LiteralPath (Join-Path $artifactPath 'training-summary.json') -PathType Leaf)) { $issues.Add('formal_training_state') }
    if ([string]$readiness.source_v24_initial_model_state_sha256 -ne 'fecd0ca1aa21c0a9cdb4b0425c64703620cb2ea0461d37e7fa5a2057cea8c877') { $issues.Add('v24_initialization') }
    if ([string]::IsNullOrWhiteSpace([string]$readiness.paired_v21_seed43_baseline_initial_model_state_sha256)) { $issues.Add('v21_reference_initialization') }
    foreach ($property in @($readiness.checks.PSObject.Properties)) {
        if ([bool]$property.Value -ne $true) { $issues.Add('check:' + $property.Name) }
    }
    if ($issues.Count -ne 0) { throw ('V25a readiness is stale or failed: ' + ($issues -join ',') + '; path=' + $readinessPath) }
}

function Assert-GpuMemoryHeadroom {
    $nvidiaSmi = Get-Command 'nvidia-smi.exe' -ErrorAction SilentlyContinue
    if ($null -eq $nvidiaSmi) { throw 'nvidia-smi.exe is unavailable; V25a training requires a CUDA headroom audit.' }
    $output = @(& $nvidiaSmi.Source @('--id=0', '--query-gpu=memory.free,memory.total', '--format=csv,noheader,nounits'))
    if ([int]$LASTEXITCODE -ne 0 -or $output.Count -ne 1) { throw 'Unable to query GPU 0 memory with nvidia-smi.' }
    $freeMiB = [int]::Parse((([string]$output[0] -split ',')[0]).Trim(), [System.Globalization.CultureInfo]::InvariantCulture)
    Write-Host ('[V25a] GPU0 free=' + $freeMiB + ' MiB; required=' + $minimumFreeGpuMemoryMiB + ' MiB') -ForegroundColor Cyan
    if ($freeMiB -lt $minimumFreeGpuMemoryMiB) { throw ('V25a CUDA preflight blocked: free GPU memory is below ' + $minimumFreeGpuMemoryMiB + ' MiB.') }
}

function Assert-CompletedAudit {
    $summaryPath = Join-Path $artifactPath 'training-summary.json'
    Assert-File $summaryPath 'V25a training summary'
    Assert-File (Join-Path $artifactPath 'best_model.pt') 'V25a best checkpoint'
    Assert-File (Join-Path $artifactPath 'last_model.pt') 'V25a last checkpoint'
    Assert-File (Join-Path $artifactPath 'sample_detail_information.json') 'V25a sample detail'
    Assert-File (Join-Path $artifactPath 'gradient-telemetry\metadata.json') 'V25a telemetry metadata'
    for ($epoch = 1; $epoch -le 20; $epoch++) {
        Assert-File (Join-Path $artifactPath ('epochs\epoch-' + $epoch.ToString('000') + '.pt')) ('V25a epoch checkpoint ' + $epoch)
        Assert-File (Join-Path $artifactPath ('gradient-telemetry\epoch-' + $epoch.ToString('000') + '.json')) ('V25a telemetry epoch ' + $epoch)
    }
    $summary = Read-Utf8Json $summaryPath
    if ([string]$summary.status -ne 'completed' -or [string]$summary.variant -ne 'v25a_course_preclip_gradient_telemetry') { throw ('V25a completed artifact identity is invalid: ' + $summaryPath) }
    if (@($summary.history).Count -ne 20 -or [int]$summary.global_step -ne 9860 -or $summary.full_parameter_training -ne $true) { throw 'V25a full 20-epoch telemetry training contract is invalid.' }
    if ($summary.paired_initialization.sealed_v24_initial_model_state_reproduced -ne $true -or $summary.paired_initialization.gradient_telemetry_only -ne $true) { throw 'V25a fresh initialization or telemetry-only audit is missing.' }
    if ([string]$summary.initial_model_state_sha256 -ne 'fecd0ca1aa21c0a9cdb4b0425c64703620cb2ea0461d37e7fa5a2057cea8c877') { throw 'V25a initial state does not reproduce V24.' }
    if ([int]$summary.gradient_telemetry.record_count -ne 9860 -or [int]$summary.gradient_telemetry.epoch_file_count -ne 20) { throw 'V25a telemetry completeness is invalid.' }
    if ($summary.test_evaluation -ne 'not_executed' -or $summary.deployment_eligible -ne $false) { throw 'V25a completed scope is not sealed.' }
}

function Archive-NonResumableFailedArtifact {
    if (-not (Test-Path -LiteralPath $artifactPath -PathType Container)) { return $false }
    $summaryPath = Join-Path $artifactPath 'training-summary.json'
    $lastCheckpoint = Join-Path $artifactPath 'last_model.pt'
    $epochPath = Join-Path $artifactPath 'epochs'
    if (-not (Test-Path -LiteralPath $summaryPath -PathType Leaf)) { return $false }
    if (Test-Path -LiteralPath $lastCheckpoint -PathType Leaf) { return $false }
    if (Test-Path -LiteralPath $epochPath -PathType Container) {
        $epochCheckpoints = @(Get-ChildItem -LiteralPath $epochPath -Filter '*.pt' -File)
        if ($epochCheckpoints.Count -ne 0) { return $false }
    }
    $summary = Read-Utf8Json $summaryPath
    if ([string]$summary.status -ne 'failed') { return $false }
    $timestamp = (Get-Date).ToString('yyyyMMdd-HHmmss-fff', [System.Globalization.CultureInfo]::InvariantCulture)
    $archivePath = $artifactPath + '.failed-before-checkpoint-' + $timestamp
    if (Test-Path -LiteralPath $archivePath) { throw ('V25a failed-artifact archive path already exists: ' + $archivePath) }
    Move-Item -LiteralPath $artifactPath -Destination $archivePath
    Write-Host ('[V25a] archived non-resumable failed attempt to ' + $archivePath) -ForegroundColor Yellow
    return $true
}

function Invoke-Prepare {
    Invoke-RadarPython @('-m', 'py_compile', 'radar_ln3\known_gate_features.py', 'radar_ln3\known_gate.py', 'radar_ln3\known_gate_gradient_telemetry.py', 'build\known_gate_v25a.py', 'train\known_gate_v25a.py', 'report\known_gate_v25a.py')
    Invoke-RadarPython @('-m', 'build.known_gate_v25a', '--project-root', $projectPath)
    Assert-Readiness
}


function Invoke-Train {
    Assert-Readiness
    $summaryPath = Join-Path $artifactPath 'training-summary.json'
    $lastCheckpoint = Join-Path $artifactPath 'last_model.pt'
    if (Test-Path -LiteralPath $summaryPath -PathType Leaf) {
        $summary = Read-Utf8Json $summaryPath
        if ([string]$summary.status -eq 'completed') { Assert-CompletedAudit; Write-Host '[V25a] completed telemetry audit exists; preserving it.' -ForegroundColor Yellow; return }
    }
    $arguments = @('-m', 'train.known_gate_v25a', '--project-root', $projectPath)
    if (Test-Path -LiteralPath $artifactPath -PathType Container) {
        $items = @(Get-ChildItem -LiteralPath $artifactPath -Force)
        if ($items.Count -gt 0) {
            if (Test-Path -LiteralPath $lastCheckpoint -PathType Leaf) { $arguments += @('--resume-checkpoint', $lastCheckpoint); Write-Host ('[V25a] resuming from ' + $lastCheckpoint) -ForegroundColor Yellow }
            elseif (-not (Archive-NonResumableFailedArtifact)) { throw ('V25a artifact has no resumable last_model.pt; preserve it for inspection before retrying: ' + $artifactPath) }
        }
    }
    Assert-GpuMemoryHeadroom
    Invoke-RadarPython $arguments
    Assert-CompletedAudit
}

function Invoke-Evaluate {
    Assert-CompletedAudit
    Invoke-RadarPython @('-m', 'report.known_gate_v25a', '--project-root', $projectPath)
    Assert-File $reportPath 'V25a telemetry audit report'
    $report = Read-Utf8Json $reportPath
    if ([string]$report.status -ne 'completed_development_gradient_telemetry_audit_only' -or [string]$report.schema_version -ne 'RADAR-LN3-V25a-known-gate-course-preclip-gradient-telemetry-audit-v1' -or $report.deployment_eligible -ne $false -or [int]$report.integrity.record_count -ne 9860) { throw ('V25a report violates sealed telemetry scope: ' + $reportPath) }
}

try {
    Assert-File $python 'Project Python interpreter'
    Write-Host ('[V25a] project=' + $projectPath) -ForegroundColor Green
    Write-Host ('[V25a] stage=' + $Stage + '; V24-equivalent full20epoch pre-clip gradient telemetry audit') -ForegroundColor Green
    Write-Host '[V25a] Only read-only telemetry is added. No OOF, test, Unknown, L3, full cascade, rejection, stable or P95 is executed.' -ForegroundColor Green
    switch ($Stage) {
        'prepare' { Invoke-Prepare }
        'train' { Invoke-Train }
        'evaluate' { Invoke-Evaluate }
        'all' { Invoke-Prepare; Invoke-Train; Invoke-Evaluate }
    }
    Write-Host '[V25a] requested stage completed.' -ForegroundColor Green
}
catch { [Console]::Error.WriteLine('[V25a] pipeline stopped: ' + $_.Exception.Message); exit 1 }
