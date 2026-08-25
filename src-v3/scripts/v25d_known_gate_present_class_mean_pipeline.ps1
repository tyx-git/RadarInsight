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
$artifactPath = Join-Path $projectPath 'artifacts\RADAR-LN3-V25d-cq08-liquid-known-gate-course-present-class-mean-seed43-formal'
$readinessPath = Join-Path $projectPath 'src-v3\processed\report\RADAR-LN3-V25d-known-gate-course-present-class-mean-training-readiness.json'
$reportPath = Join-Path $projectPath 'src-v3\processed\report\RADAR-LN3-V25d-known-gate-course-present-class-mean-paired-development.json'
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
    Assert-File $readinessPath 'V25d readiness'
    $readiness = Read-Utf8Json $readinessPath
    $issues = [System.Collections.Generic.List[string]]::new()
    if ([string]$readiness.status -ne 'passed') { $issues.Add('status') }
    if ([string]$readiness.schema_version -ne 'RADAR-LN3-V25d-known-gate-course-present-class-mean-training-readiness-v1') { $issues.Add('schema_version') }
    if ([string]$readiness.variant -ne 'v25d_course_present_class_mean') { $issues.Add('variant') }
    if ([int]$readiness.known_train_parent_count -ne 986 -or [int]$readiness.known_validation_parent_count -ne 232 -or [int]$readiness.validation_clutter_parent_support -ne 9) { $issues.Add('support') }
    if ([int]$readiness.unknown_training_exposure -ne 0) { $issues.Add('unknown_training_exposure') }
    if ([bool]$readiness.validation_used_for_readiness_gate) { $issues.Add('validation_used_for_readiness_gate') }
    if ([int]$readiness.test_payload_access_count -ne 0 -or [bool]$readiness.deployment_eligible) { $issues.Add('sealed_scope') }
    if ([string]$readiness.formal_training -ne 'not_executed' -and -not (Test-Path -LiteralPath (Join-Path $artifactPath 'training-summary.json') -PathType Leaf)) { $issues.Add('formal_training_state') }
    if ([string]$readiness.source_v25a_initial_model_state_sha256 -ne 'fecd0ca1aa21c0a9cdb4b0425c64703620cb2ea0461d37e7fa5a2057cea8c877') { $issues.Add('v25a_initialization') }
    if ((@($readiness.loss_semantics_audit.mixed_final_effective_class_mass) -join ',') -ne '0.5,0.5') { $issues.Add('final_mass') }
    if ((@($readiness.loss_semantics_audit.mixed_prefix_effective_class_mass) -join ',') -ne '0.5,0.5') { $issues.Add('prefix_mass') }
    foreach ($property in @($readiness.checks.PSObject.Properties)) {
        if ([bool]$property.Value -ne $true) { $issues.Add('check:' + $property.Name) }
    }
    if ($issues.Count -ne 0) { throw ('V25d readiness is stale or failed: ' + ($issues -join ',') + '; path=' + $readinessPath) }
}

function Assert-GpuMemoryHeadroom {
    $nvidiaSmi = Get-Command 'nvidia-smi.exe' -ErrorAction SilentlyContinue
    if ($null -eq $nvidiaSmi) { throw 'nvidia-smi.exe is unavailable; V25d training requires a CUDA headroom audit.' }
    $output = @(& $nvidiaSmi.Source @('--id=0', '--query-gpu=memory.free,memory.total', '--format=csv,noheader,nounits'))
    if ([int]$LASTEXITCODE -ne 0 -or $output.Count -ne 1) { throw 'Unable to query GPU 0 memory with nvidia-smi.' }
    $freeMiB = [int]::Parse((([string]$output[0] -split ',')[0]).Trim(), [System.Globalization.CultureInfo]::InvariantCulture)
    Write-Host ('[V25d] GPU0 free=' + $freeMiB + ' MiB; required=' + $minimumFreeGpuMemoryMiB + ' MiB') -ForegroundColor Cyan
    if ($freeMiB -lt $minimumFreeGpuMemoryMiB) { throw ('V25d CUDA preflight blocked: free GPU memory is below ' + $minimumFreeGpuMemoryMiB + ' MiB.') }
}

function Assert-CompletedTraining {
    $summaryPath = Join-Path $artifactPath 'training-summary.json'
    Assert-File $summaryPath 'V25d training summary'
    Assert-File (Join-Path $artifactPath 'best_model.pt') 'V25d best checkpoint'
    Assert-File (Join-Path $artifactPath 'last_model.pt') 'V25d last checkpoint'
    Assert-File (Join-Path $artifactPath 'sample_detail_information.json') 'V25d sample detail'
    for ($epoch = 1; $epoch -le 20; $epoch++) { Assert-File (Join-Path $artifactPath ('epochs\epoch-' + $epoch.ToString('000') + '.pt')) ('V25d epoch checkpoint ' + $epoch) }
    $summary = Read-Utf8Json $summaryPath
    if ([string]$summary.status -ne 'completed' -or [string]$summary.variant -ne 'v25d_course_present_class_mean') { throw ('V25d completed artifact identity is invalid: ' + $summaryPath) }
    if (@($summary.history).Count -ne 20 -or [int]$summary.global_step -ne 9860 -or $summary.full_parameter_training -ne $true) { throw 'V25d full 20-epoch training contract is invalid.' }
    if ($summary.paired_initialization.sealed_v25a_initial_model_state_reproduced -ne $true -or $summary.paired_initialization.present_class_mean_loss_only_pair -ne $true) { throw 'V25d fresh loss-only paired initialization is missing.' }
    if ([string]$summary.initial_model_state_sha256 -ne 'fecd0ca1aa21c0a9cdb4b0425c64703620cb2ea0461d37e7fa5a2057cea8c877') { throw 'V25d initial state does not reproduce V25a.' }
    if ([string]$summary.present_class_mean_loss_mass_telemetry.normalization -ne 'present_class_mean') { throw 'V25d loss normalization telemetry is missing.' }
    $final = $summary.present_class_mean_loss_mass_telemetry.by_scope.final
    if ([int]$final.contributing_batch_count -ne 9860 -or [int]$final.mixed_batch_count -le 0 -or $final.mixed_batches_exactly_equal_class_mass -ne $true -or $final.pure_batches_have_single_class_mass -ne $true) { throw 'V25d final loss-class-mass telemetry is incomplete.' }
    if ($summary.test_evaluation -ne 'not_executed' -or $summary.deployment_eligible -ne $false) { throw 'V25d completed scope is not sealed.' }
}

function Archive-NonResumableFailedArtifact {
    if (-not (Test-Path -LiteralPath $artifactPath -PathType Container)) { return $false }
    $summaryPath = Join-Path $artifactPath 'training-summary.json'
    $lastCheckpoint = Join-Path $artifactPath 'last_model.pt'
    if (-not (Test-Path -LiteralPath $summaryPath -PathType Leaf) -or (Test-Path -LiteralPath $lastCheckpoint -PathType Leaf)) { return $false }
    $summary = Read-Utf8Json $summaryPath
    if ([string]$summary.status -ne 'failed') { return $false }
    $timestamp = (Get-Date).ToString('yyyyMMdd-HHmmss-fff', [System.Globalization.CultureInfo]::InvariantCulture)
    $archivePath = $artifactPath + '.failed-before-checkpoint-' + $timestamp
    if (Test-Path -LiteralPath $archivePath) { throw ('V25d failed-artifact archive path already exists: ' + $archivePath) }
    Move-Item -LiteralPath $artifactPath -Destination $archivePath
    Write-Host ('[V25d] archived non-resumable failed attempt to ' + $archivePath) -ForegroundColor Yellow
    return $true
}

function Invoke-Prepare {
    Invoke-RadarPython @('-m', 'py_compile', 'radar_ln3\losses.py', 'radar_ln3\known_gate_features.py', 'radar_ln3\known_gate.py', 'build\known_gate_v25d.py', 'train\known_gate_v21.py', 'train\known_gate_v25d.py', 'report\known_gate_v25d.py', 'smoke\known_gate_v25d_contract_test.py')
    Invoke-RadarPython @('-m', 'build.known_gate_v25d', '--project-root', $projectPath)
    Assert-Readiness
}

function Invoke-Smoke {
    Assert-Readiness
    Invoke-RadarPython @('-m', 'smoke.known_gate_v25d_contract_test', '--project-root', $projectPath)
    Invoke-RadarPython @('-m', 'smoke.static_isolation_check', '--root', $srcRoot)
}

function Invoke-Train {
    Assert-Readiness
    $summaryPath = Join-Path $artifactPath 'training-summary.json'
    $lastCheckpoint = Join-Path $artifactPath 'last_model.pt'
    if (Test-Path -LiteralPath $summaryPath -PathType Leaf) {
        $summary = Read-Utf8Json $summaryPath
        if ([string]$summary.status -eq 'completed') { Assert-CompletedTraining; Write-Host '[V25d] completed training exists; preserving it.' -ForegroundColor Yellow; return }
    }
    $arguments = @('-m', 'train.known_gate_v25d', '--project-root', $projectPath)
    if (Test-Path -LiteralPath $artifactPath -PathType Container) {
        $items = @(Get-ChildItem -LiteralPath $artifactPath -Force)
        if ($items.Count -gt 0) {
            if (Test-Path -LiteralPath $lastCheckpoint -PathType Leaf) { $arguments += @('--resume-checkpoint', $lastCheckpoint); Write-Host ('[V25d] resuming from ' + $lastCheckpoint) -ForegroundColor Yellow }
            elseif (-not (Archive-NonResumableFailedArtifact)) { throw ('V25d artifact has no resumable last_model.pt; preserve it for inspection before retrying: ' + $artifactPath) }
        }
    }
    Assert-GpuMemoryHeadroom
    Invoke-RadarPython $arguments
    Assert-CompletedTraining
}

function Invoke-Evaluate {
    Assert-CompletedTraining
    Invoke-RadarPython @('-m', 'report.known_gate_v25d', '--project-root', $projectPath)
    Assert-File $reportPath 'V25d paired development report'
    $report = Read-Utf8Json $reportPath
    if ([string]$report.status -ne 'completed_development_paired_loss_only_experiment' -or [string]$report.schema_version -ne 'RADAR-LN3-V25d-known-gate-course-present-class-mean-paired-development-v1' -or $report.deployment_eligible -ne $false -or $report.loss_mass_mechanism.by_scope.final.mixed_batches_exactly_equal_class_mass -ne $true) { throw ('V25d report violates sealed loss-only scope: ' + $reportPath) }
}

try {
    Assert-File $python 'Project Python interpreter'
    Write-Host ('[V25d] project=' + $projectPath) -ForegroundColor Green
    Write-Host ('[V25d] stage=' + $Stage + '; V25a paired present-class-mean mixed-batch loss-only treatment; full20epoch') -ForegroundColor Green
    Write-Host '[V25d] No OOF, test, Unknown, L3, full cascade, rejection, stable or P95 is executed.' -ForegroundColor Green
    switch ($Stage) {
        'prepare' { Invoke-Prepare }
        'smoke' { Invoke-Smoke }
        'train' { Invoke-Train }
        'evaluate' { Invoke-Evaluate }
        'all' { Invoke-Prepare; Invoke-Smoke; Invoke-Train; Invoke-Evaluate }
    }
    Write-Host '[V25d] requested stage completed.' -ForegroundColor Green
}
catch { [Console]::Error.WriteLine('[V25d] pipeline stopped: ' + $_.Exception.Message); exit 1 }
