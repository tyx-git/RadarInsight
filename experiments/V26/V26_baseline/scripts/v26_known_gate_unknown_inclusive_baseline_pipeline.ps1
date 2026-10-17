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
$artifactPath = Join-Path $projectPath 'artifacts\RADAR-LN3-V26-cq08-liquid-known-gate-known-only-open-set-baseline-seed43-formal'
$readinessPath = Join-Path $projectPath 'src-v3\processed\report\RADAR-LN3-V26-known-gate-known-only-open-set-baseline-training-readiness.json'
$reportPath = Join-Path $projectPath 'src-v3\processed\report\RADAR-LN3-V26-known-gate-known-only-open-set-development.json'
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
    Assert-File $readinessPath 'V26 readiness'
    $readiness = Read-Utf8Json $readinessPath
    $issues = [System.Collections.Generic.List[string]]::new()
    if ([string]$readiness.status -ne 'passed') { $issues.Add('status') }
    if ([string]$readiness.schema_version -ne 'RADAR-LN3-V26-known-gate-known-only-open-set-baseline-training-readiness-v1') { $issues.Add('schema_version') }
    if ([string]$readiness.variant -ne 'v26_known_only_open_set_baseline') { $issues.Add('variant') }
    if ([int]$readiness.known_train_parent_count -ne 986 -or [int]$readiness.known_validation_parent_count -ne 232) { $issues.Add('known_support') }
    if ([int]$readiness.audit_validation_parent_count -ne 247 -or [int]$readiness.audit_validation_unknown_parent_count -ne 15) { $issues.Add('validation_audit_support') }
    if ([int]$readiness.planned_audit_test_parent_count -ne 244 -or [int]$readiness.planned_audit_test_unknown_parent_count -ne 16) { $issues.Add('test_audit_support') }
    if ([int]$readiness.unknown_training_exposure -ne 0) { $issues.Add('unknown_training_exposure') }
    if ([string]$readiness.initial_model_state_sha256 -notmatch '^[0-9a-fA-F]{64}$') { $issues.Add('initial_model_state_sha256') }
    if ([bool]$readiness.validation_used_for_readiness_gate) { $issues.Add('validation_used_for_readiness_gate') }
    if ([int]$readiness.test_payload_access_count -ne 0 -or [bool]$readiness.deployment_eligible) { $issues.Add('sealed_scope') }
    foreach ($property in @($readiness.checks.PSObject.Properties)) {
        if ([bool]$property.Value -ne $true) { $issues.Add('check:' + $property.Name) }
    }
    if ($issues.Count -ne 0) { throw ('V26 readiness is stale or failed: ' + ($issues -join ',') + '; path=' + $readinessPath) }
}

function Assert-GpuMemoryHeadroom {
    $nvidiaSmi = Get-Command 'nvidia-smi.exe' -ErrorAction SilentlyContinue
    if ($null -eq $nvidiaSmi) { throw 'nvidia-smi.exe is unavailable; V26 training requires a CUDA headroom audit.' }
    $output = @(& $nvidiaSmi.Source @('--id=0', '--query-gpu=memory.free,memory.total', '--format=csv,noheader,nounits'))
    if ([int]$LASTEXITCODE -ne 0 -or $output.Count -ne 1) { throw 'Unable to query GPU 0 memory with nvidia-smi.' }
    $freeMiB = [int]::Parse((([string]$output[0] -split ',')[0]).Trim(), [System.Globalization.CultureInfo]::InvariantCulture)
    Write-Host ('[V26] GPU0 free=' + $freeMiB + ' MiB; required=' + $minimumFreeGpuMemoryMiB) -ForegroundColor Cyan
    if ($freeMiB -lt $minimumFreeGpuMemoryMiB) { throw ('V26 CUDA preflight blocked: free GPU memory is below ' + $minimumFreeGpuMemoryMiB + ' MiB.') }
}

function Assert-CompletedBaseline {
    $summaryPath = Join-Path $artifactPath 'training-summary.json'
    Assert-File $summaryPath 'V26 training summary'
    Assert-File (Join-Path $artifactPath 'best_model.pt') 'V26 best checkpoint'
    Assert-File (Join-Path $artifactPath 'last_model.pt') 'V26 last checkpoint'
    Assert-File (Join-Path $artifactPath 'sample_detail_information.json') 'V26 sample detail'
    for ($epoch = 1; $epoch -le 20; $epoch++) {
        Assert-File (Join-Path $artifactPath ('epochs\epoch-' + $epoch.ToString('000') + '.pt')) ('V26 epoch ' + $epoch)
    }
    $summary = Read-Utf8Json $summaryPath
    if ([string]$summary.status -ne 'completed' -or [string]$summary.variant -ne 'v26_known_only_open_set_baseline') { throw ('V26 completed artifact identity is invalid: ' + $summaryPath) }
    if (@($summary.history).Count -ne 20 -or $summary.full_parameter_training -ne $true) { throw 'V26 full 20-epoch training contract is invalid.' }
    if ($summary.unknown_is_model_class -ne $false -or $summary.test_evaluation -ne 'not_executed' -or $summary.deployment_eligible -ne $false) { throw 'V26 training scope is not sealed as Known-only.' }
}

function Invoke-Prepare {
    Invoke-RadarPython @('-m', 'py_compile', 'radar_ln3\data.py', 'radar_ln3\known_gate_features.py', 'radar_ln3\known_gate.py', 'radar_ln3\known_gate_v26_audit.py', 'build\known_gate_v26.py', 'train\known_gate_common.py', 'train\known_gate_v26.py', 'report\known_gate_v26.py')
    Invoke-RadarPython @('-m', 'build.known_gate_v26', '--project-root', $projectPath)
    Assert-Readiness
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
    if (Test-Path -LiteralPath $archivePath) { throw ('V26 failed-artifact archive path already exists: ' + $archivePath) }
    Move-Item -LiteralPath $artifactPath -Destination $archivePath
    Write-Host ('[V26] archived non-resumable failed attempt to ' + $archivePath) -ForegroundColor Yellow
    return $true
}

function Invoke-Train {
    Assert-Readiness
    $summaryPath = Join-Path $artifactPath 'training-summary.json'
    $lastCheckpoint = Join-Path $artifactPath 'last_model.pt'
    if (Test-Path -LiteralPath $summaryPath -PathType Leaf) {
        $summary = Read-Utf8Json $summaryPath
        if ([string]$summary.status -eq 'completed') { Assert-CompletedBaseline; Write-Host '[V26] completed Known-only baseline exists; preserving it.' -ForegroundColor Yellow; return }
    }
    $arguments = @('-m', 'train.known_gate_v26', '--project-root', $projectPath)
    if (Test-Path -LiteralPath $artifactPath -PathType Container) {
        $items = @(Get-ChildItem -LiteralPath $artifactPath -Force)
        if ($items.Count -gt 0) {
            if (Test-Path -LiteralPath $lastCheckpoint -PathType Leaf) { $arguments += @('--resume-checkpoint', $lastCheckpoint); Write-Host ('[V26] resuming from ' + $lastCheckpoint) -ForegroundColor Yellow }
            elseif (-not (Archive-NonResumableFailedArtifact)) { throw ('V26 artifact has no resumable last_model.pt; preserve it for inspection before retrying: ' + $artifactPath) }
        }
    }
    Assert-GpuMemoryHeadroom
    Invoke-RadarPython $arguments
    Assert-CompletedBaseline
}

function Invoke-Evaluate {
    Assert-CompletedBaseline
    Assert-GpuMemoryHeadroom
    Invoke-RadarPython @('-m', 'report.known_gate_v26', '--project-root', $projectPath)
    Assert-File $reportPath 'V26 development report'
    $report = Read-Utf8Json $reportPath
    if ([string]$report.status -ne 'completed_development_open_set_audit' -or [bool]$report.deployment_eligible) { throw ('V26 report violates sealed scope: ' + $reportPath) }
}

try {
    Assert-File $python 'Project Python interpreter'
    Write-Host ('[V26] project=' + $projectPath) -ForegroundColor Green
    Write-Host ('[V26] stage=' + $Stage + '; Known-only binary training plus post-checkpoint mixed open-set audit') -ForegroundColor Green
    Write-Host '[V26] UnknownTarget is excluded from training, normalization and checkpoint ranking. Mixed val/test audit is only after the frozen best checkpoint. L2/L3, full cascade, rejection, stable, P95 and deployment remain not_executed.' -ForegroundColor Green
    switch ($Stage) {
        'prepare' { Invoke-Prepare }
        'train' { Invoke-Train }
        'evaluate' { Invoke-Evaluate }
        'all' { Invoke-Prepare; Invoke-Train; Invoke-Evaluate }
    }
    Write-Host '[V26] requested stage completed.' -ForegroundColor Green
}
catch { [Console]::Error.WriteLine('[V26] pipeline stopped: ' + $_.Exception.Message); exit 1 }
