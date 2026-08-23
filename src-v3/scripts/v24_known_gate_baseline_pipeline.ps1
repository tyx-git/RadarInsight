[CmdletBinding()]
param(
    [ValidateSet('prepare', 'smoke', 'train', 'evaluate', 'all')]
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
$artifactPath = Join-Path $projectPath 'artifacts\RADAR-LN3-V24-cq08-liquid-known-gate-course-reason-aware-baseline-seed43-formal'
$readinessPath = Join-Path $projectPath 'src-v3\processed\report\RADAR-LN3-V24-known-gate-course-reason-aware-baseline-training-readiness.json'
$reportPath = Join-Path $projectPath 'src-v3\processed\report\RADAR-LN3-V24-known-gate-course-reason-aware-baseline-development.json'
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
    Assert-File $readinessPath 'V24 readiness'
    $readiness = Read-Utf8Json $readinessPath
    $issues = [System.Collections.Generic.List[string]]::new()
    if ([string]$readiness.status -ne 'passed') { $issues.Add('status') }
    if ([string]$readiness.schema_version -ne 'RADAR-LN3-V24-known-gate-course-reason-aware-baseline-training-readiness-v1') { $issues.Add('schema_version') }
    if ([string]$readiness.variant -ne 'v24_course_reason_aware_baseline') { $issues.Add('variant') }
    if ([int]$readiness.known_train_parent_count -ne 986 -or [int]$readiness.known_validation_parent_count -ne 232 -or [int]$readiness.validation_clutter_parent_support -ne 9) { $issues.Add('support') }
    if ([int]$readiness.unknown_training_exposure -ne 0) { $issues.Add('unknown_training_exposure') }
    if ([bool]$readiness.validation_used_for_readiness_gate) { $issues.Add('validation_used_for_readiness_gate') }
    if ([int]$readiness.test_payload_access_count -ne 0 -or [bool]$readiness.deployment_eligible) { $issues.Add('sealed_scope') }
    if ([string]$readiness.formal_training -ne 'not_executed' -and -not (Test-Path -LiteralPath (Join-Path $artifactPath 'training-summary.json') -PathType Leaf)) { $issues.Add('formal_training_state') }
    foreach ($property in @($readiness.checks.PSObject.Properties)) {
        if ([bool]$property.Value -ne $true) { $issues.Add('check:' + $property.Name) }
    }
    if ($issues.Count -ne 0) { throw ('V24 readiness is stale or failed: ' + ($issues -join ',') + '; path=' + $readinessPath) }
}

function Assert-GpuMemoryHeadroom {
    $nvidiaSmi = Get-Command 'nvidia-smi.exe' -ErrorAction SilentlyContinue
    if ($null -eq $nvidiaSmi) { throw 'nvidia-smi.exe is unavailable; V24 training requires a CUDA headroom audit.' }
    $output = @(& $nvidiaSmi.Source @('--id=0', '--query-gpu=memory.free,memory.total', '--format=csv,noheader,nounits'))
    if ([int]$LASTEXITCODE -ne 0 -or $output.Count -ne 1) { throw 'Unable to query GPU 0 memory with nvidia-smi.' }
    $freeMiB = [int]::Parse((([string]$output[0] -split ',')[0]).Trim(), [System.Globalization.CultureInfo]::InvariantCulture)
    Write-Host ('[V24] GPU0 free=' + $freeMiB + ' MiB; required=' + $minimumFreeGpuMemoryMiB + ' MiB') -ForegroundColor Cyan
    if ($freeMiB -lt $minimumFreeGpuMemoryMiB) { throw ('V24 CUDA preflight blocked: free GPU memory is below ' + $minimumFreeGpuMemoryMiB + ' MiB.') }
}

function Assert-CompletedBaseline {
    $summaryPath = Join-Path $artifactPath 'training-summary.json'
    Assert-File $summaryPath 'V24 training summary'
    Assert-File (Join-Path $artifactPath 'best_model.pt') 'V24 best checkpoint'
    Assert-File (Join-Path $artifactPath 'last_model.pt') 'V24 last checkpoint'
    Assert-File (Join-Path $artifactPath 'sample_detail_information.json') 'V24 sample detail'
    for ($epoch = 1; $epoch -le 20; $epoch++) {
        Assert-File (Join-Path $artifactPath ('epochs\epoch-' + $epoch.ToString('000') + '.pt')) ('V24 epoch ' + $epoch)
    }
    $summary = Read-Utf8Json $summaryPath
    if ([string]$summary.status -ne 'completed' -or [string]$summary.variant -ne 'v24_course_reason_aware_baseline') { throw ('V24 completed artifact identity is invalid: ' + $summaryPath) }
    if (@($summary.history).Count -ne 20 -or $summary.full_parameter_training -ne $true) { throw 'V24 full 20-epoch training contract is invalid.' }
    if ($summary.paired_initialization.sealed_v24_initial_model_state_reproduced -ne $true) { throw 'V24 fresh initialization audit is missing from the completed summary.' }
    if ($summary.test_evaluation -ne 'not_executed' -or $summary.deployment_eligible -ne $false) { throw 'V24 completed scope is not sealed.' }
}

function Invoke-Prepare {
    Invoke-RadarPython @('-m', 'py_compile', 'radar_ln3\known_gate_features.py', 'radar_ln3\known_gate.py', 'build\known_gate_v24.py', 'train\known_gate_v24.py', 'report\known_gate_v24.py', 'smoke\known_gate_v24_contract_test.py')
    Invoke-RadarPython @('-m', 'build.known_gate_v24', '--project-root', $projectPath)
    Assert-Readiness
}

function Invoke-Smoke {
    Assert-Readiness
    Invoke-RadarPython @('-m', 'smoke.known_gate_v24_contract_test', '--project-root', $projectPath)
    Invoke-RadarPython @('-m', 'smoke.static_isolation_check', '--root', $srcRoot)
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
    if (Test-Path -LiteralPath $archivePath) { throw ('V24 failed-artifact archive path already exists: ' + $archivePath) }
    Move-Item -LiteralPath $artifactPath -Destination $archivePath
    Write-Host ('[V24] archived non-resumable failed attempt to ' + $archivePath) -ForegroundColor Yellow
    return $true
}

function Invoke-Train {
    Assert-Readiness
    $summaryPath = Join-Path $artifactPath 'training-summary.json'
    $lastCheckpoint = Join-Path $artifactPath 'last_model.pt'
    if (Test-Path -LiteralPath $summaryPath -PathType Leaf) {
        $summary = Read-Utf8Json $summaryPath
        if ([string]$summary.status -eq 'completed') { Assert-CompletedBaseline; Write-Host '[V24] completed baseline exists; preserving it.' -ForegroundColor Yellow; return }
    }
    $arguments = @('-m', 'train.known_gate_v24', '--project-root', $projectPath)
    if (Test-Path -LiteralPath $artifactPath -PathType Container) {
        $items = @(Get-ChildItem -LiteralPath $artifactPath -Force)
        if ($items.Count -gt 0) {
            if (Test-Path -LiteralPath $lastCheckpoint -PathType Leaf) { $arguments += @('--resume-checkpoint', $lastCheckpoint); Write-Host ('[V24] resuming from ' + $lastCheckpoint) -ForegroundColor Yellow }
            elseif (-not (Archive-NonResumableFailedArtifact)) { throw ('V24 artifact has no resumable last_model.pt; preserve it for inspection before retrying: ' + $artifactPath) }
        }
    }
    Assert-GpuMemoryHeadroom
    Invoke-RadarPython $arguments
    Assert-CompletedBaseline
}

function Invoke-Evaluate {
    Assert-CompletedBaseline
    Invoke-RadarPython @('-m', 'report.known_gate_v24', '--project-root', $projectPath)
    Assert-File $reportPath 'V24 development report'
    $report = Read-Utf8Json $reportPath
    if ([string]$report.status -ne 'completed_development_baseline_only' -or [string]$report.schema_version -ne 'RADAR-LN3-V24-known-gate-course-reason-aware-baseline-development-v1' -or $report.deployment_eligible -ne $false) { throw ('V24 report violates sealed scope: ' + $reportPath) }
}

try {
    Assert-File $python 'Project Python interpreter'
    Write-Host ('[V24] project=' + $projectPath) -ForegroundColor Green
    Write-Host ('[V24] stage=' + $Stage + '; fresh V21e-style self-developed KnownGate baseline, full20epoch') -ForegroundColor Green
    Write-Host '[V24] V21e course cache is reused; trained V21e/V21 checkpoints are not loaded. OOF, test, Unknown, L3, full cascade, rejection, stable and P95 remain not_executed.' -ForegroundColor Green
    switch ($Stage) {
        'prepare' { Invoke-Prepare }
        'smoke' { Invoke-Smoke }
        'train' { Invoke-Train }
        'evaluate' { Invoke-Evaluate }
        'all' { Invoke-Prepare; Invoke-Smoke; Invoke-Train; Invoke-Evaluate }
    }
    Write-Host '[V24] requested stage completed.' -ForegroundColor Green
}
catch { [Console]::Error.WriteLine('[V24] pipeline stopped: ' + $_.Exception.Message); exit 1 }
