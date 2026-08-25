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
$artifactPath = Join-Path $projectPath 'artifacts\RADAR-LN3-V25e-cq08-liquid-known-gate-course-shared-gradient-conflict-projection-seed43-formal'
$readinessPath = Join-Path $projectPath 'src-v3\processed\report\RADAR-LN3-V25e-known-gate-course-shared-gradient-conflict-projection-training-readiness.json'
$reportPath = Join-Path $projectPath 'src-v3\processed\report\RADAR-LN3-V25e-known-gate-course-shared-gradient-conflict-projection-paired-development.json'
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
    Assert-File $readinessPath 'V25e readiness'
    $readiness = Read-Utf8Json $readinessPath
    $issues = [System.Collections.Generic.List[string]]::new()
    if ([string]$readiness.status -ne 'passed') { $issues.Add('status') }
    if ([string]$readiness.schema_version -ne 'RADAR-LN3-V25e-known-gate-shared-gradient-conflict-projection-training-readiness-v1') { $issues.Add('schema_version') }
    if ([string]$readiness.variant -ne 'v25e_shared_gradient_conflict_projection') { $issues.Add('variant') }
    if ([int]$readiness.known_train_parent_count -ne 986 -or [int]$readiness.known_validation_parent_count -ne 232 -or [int]$readiness.validation_clutter_parent_support -ne 9) { $issues.Add('support') }
    if ([int]$readiness.unknown_training_exposure -ne 0) { $issues.Add('unknown_training_exposure') }
    if ([bool]$readiness.validation_used_for_readiness_gate) { $issues.Add('validation_used_for_readiness_gate') }
    if ([int]$readiness.test_payload_access_count -ne 0 -or [bool]$readiness.deployment_eligible) { $issues.Add('sealed_scope') }
    if ([string]$readiness.source_v25a_initial_model_state_sha256 -ne 'fecd0ca1aa21c0a9cdb4b0425c64703620cb2ea0461d37e7fa5a2057cea8c877') { $issues.Add('v25a_initialization') }
    if ((@($readiness.projection_mechanism.projected_modules) -join ',') -ne 'tr_encoder,fusion,liquid') { $issues.Add('projected_modules') }
    if ((@($readiness.projection_mechanism.unchanged_modules) -join ',') -ne 'rd_encoder,classifier') { $issues.Add('unchanged_modules') }
    if ([int]$readiness.projection_mechanism.expected_total_updates -ne 9860 -or [int]$readiness.projection_mechanism.expected_mixed_batch_count -ne 638 -or [int]$readiness.projection_mechanism.expected_pure_batch_count -ne 9222) { $issues.Add('batch_counts') }
    foreach ($property in @($readiness.checks.PSObject.Properties)) {
        if ([bool]$property.Value -ne $true) { $issues.Add('check:' + $property.Name) }
    }
    if ($issues.Count -ne 0) { throw ('V25e readiness is stale or failed: ' + ($issues -join ',') + '; path=' + $readinessPath) }
}

function Assert-GpuMemoryHeadroom {
    $nvidiaSmi = Get-Command 'nvidia-smi.exe' -ErrorAction SilentlyContinue
    if ($null -eq $nvidiaSmi) { throw 'nvidia-smi.exe is unavailable; V25e training requires a CUDA headroom audit.' }
    $output = @(& $nvidiaSmi.Source @('--id=0', '--query-gpu=memory.free,memory.total', '--format=csv,noheader,nounits'))
    if ([int]$LASTEXITCODE -ne 0 -or $output.Count -ne 1) { throw 'Unable to query GPU 0 memory with nvidia-smi.' }
    $freeMiB = [int]::Parse((([string]$output[0] -split ',')[0]).Trim(), [System.Globalization.CultureInfo]::InvariantCulture)
    Write-Host ('[V25e] GPU0 free=' + $freeMiB + ' MiB; required=' + $minimumFreeGpuMemoryMiB) -ForegroundColor Cyan
    if ($freeMiB -lt $minimumFreeGpuMemoryMiB) { throw ('V25e CUDA preflight blocked: free GPU memory is below ' + $minimumFreeGpuMemoryMiB + ' MiB.') }
}

function Assert-CompletedTraining {
    $summaryPath = Join-Path $artifactPath 'training-summary.json'
    Assert-File $summaryPath 'V25e training summary'
    Assert-File (Join-Path $artifactPath 'best_model.pt') 'V25e best checkpoint'
    Assert-File (Join-Path $artifactPath 'last_model.pt') 'V25e last checkpoint'
    Assert-File (Join-Path $artifactPath 'sample_detail_information.json') 'V25e sample detail'
    $temporaryFiles = @(Get-ChildItem -LiteralPath $artifactPath -Recurse -Force -File -Filter '*.tmp')
    if ($temporaryFiles.Count -ne 0) {
        throw ('V25e artifact contains incomplete temporary checkpoint/json files; refusing completion: ' + (($temporaryFiles | ForEach-Object { $_.FullName }) -join ', '))
    }
    for ($epoch = 1; $epoch -le 20; $epoch++) { Assert-File (Join-Path $artifactPath ('epochs\epoch-' + $epoch.ToString('000') + '.pt')) ('V25e epoch checkpoint ' + $epoch) }
    $summary = Read-Utf8Json $summaryPath
    if ([string]$summary.status -ne 'completed' -or [string]$summary.variant -ne 'v25e_shared_gradient_conflict_projection') { throw ('V25e completed artifact identity is invalid: ' + $summaryPath) }
    if (@($summary.history).Count -ne 20 -or [int]$summary.global_step -ne 9860 -or $summary.full_parameter_training -ne $true) { throw 'V25e full 20-epoch training contract is invalid.' }
    if ($summary.paired_initialization.sealed_v25a_initial_model_state_reproduced -ne $true -or $summary.paired_initialization.shared_gradient_projection_pair -ne $true) { throw 'V25e paired initialization evidence is missing.' }
    if ([string]$summary.shared_gradient_projection_telemetry.by_scope.batch_count -ne '9860' -or [int]$summary.shared_gradient_projection_telemetry.by_scope.mixed_known_clutter_batch_count -ne 638 -or [int]$summary.shared_gradient_projection_telemetry.by_scope.pure_batch_count -ne 9222) { throw 'V25e projection telemetry counts are incomplete.' }
    if ($summary.shared_gradient_projection_telemetry.rd_and_classifier_gradients_exactly_unchanged -ne $true) { throw 'V25e RD/classifier unchanged audit failed.' }
    $telemetry = $summary.shared_gradient_projection_telemetry
    if ($telemetry.raw_parent_identifiers_stored -ne $false) { throw 'V25e summary must not retain raw parent identifiers.' }
    if ((@($telemetry.projected_modules) -join ',') -ne 'tr_encoder,fusion,liquid' -or (@($telemetry.unchanged_modules) -join ',') -ne 'rd_encoder,classifier') { throw 'V25e module projection scope is invalid.' }
    $scope = $telemetry.by_scope
    if ([double]$scope.max_historical_loss_reconstruction_absolute_error -gt 3.0e-5) { throw 'V25e historical loss reconstruction error exceeds the sealed tolerance.' }
    foreach ($module in @('tr_encoder', 'fusion', 'liquid')) {
        if ([int]$scope.projection_count.$module -le 0) { throw ('V25e has no observed projection for module: ' + $module) }
    }
    if ($summary.test_evaluation -ne 'not_executed' -or $summary.deployment_eligible -ne $false) { throw 'V25e completed scope is not sealed.' }
}

function Assert-NoCheckpointTemps {
    if (-not (Test-Path -LiteralPath $artifactPath -PathType Container)) { return }
    $temporaryFiles = @(Get-ChildItem -LiteralPath $artifactPath -Recurse -Force -File -Filter '*.tmp')
    if ($temporaryFiles.Count -ne 0) {
        throw ('V25e found incomplete temporary files and will not resume or overwrite the artifact: ' + (($temporaryFiles | ForEach-Object { $_.FullName }) -join ', '))
    }
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
    Move-Item -LiteralPath $artifactPath -Destination $archivePath
    Write-Host ('[V25e] archived non-resumable failed attempt to ' + $archivePath) -ForegroundColor Yellow
    return $true
}

function Invoke-Prepare {
    Invoke-RadarPython @('-m', 'py_compile', 'radar_ln3\losses.py', 'radar_ln3\known_gate_features.py', 'radar_ln3\known_gate.py', 'radar_ln3\known_gate_v25b_gradient_decomposition.py', 'radar_ln3\known_gate_v25e_gradient_projection.py', 'build\known_gate_v25e.py', 'train\known_gate_v21.py', 'train\known_gate_v25e.py', 'report\known_gate_v25e.py', 'smoke\known_gate_v25e_contract_test.py')
    Invoke-RadarPython @('-m', 'build.known_gate_v25e', '--project-root', $projectPath)
    Assert-Readiness
}

function Invoke-Smoke {
    Assert-Readiness
    Invoke-RadarPython @('-m', 'smoke.known_gate_v25e_contract_test', '--project-root', $projectPath)
    Invoke-RadarPython @('-m', 'smoke.static_isolation_check', '--root', $srcRoot)
}

function Invoke-Train {
    Assert-Readiness
    Assert-NoCheckpointTemps
    $summaryPath = Join-Path $artifactPath 'training-summary.json'
    $lastCheckpoint = Join-Path $artifactPath 'last_model.pt'
    if (Test-Path -LiteralPath $summaryPath -PathType Leaf) {
        $summary = Read-Utf8Json $summaryPath
        if ([string]$summary.status -eq 'completed') { Assert-CompletedTraining; Write-Host '[V25e] completed training exists; preserving it.' -ForegroundColor Yellow; return }
    }
    $arguments = @('-m', 'train.known_gate_v25e', '--project-root', $projectPath)
    if (Test-Path -LiteralPath $artifactPath -PathType Container) {
        $items = @(Get-ChildItem -LiteralPath $artifactPath -Force)
        if ($items.Count -gt 0) {
            if (Test-Path -LiteralPath $lastCheckpoint -PathType Leaf) { $arguments += @('--resume-checkpoint', $lastCheckpoint); Write-Host ('[V25e] resuming from ' + $lastCheckpoint) -ForegroundColor Yellow }
            elseif (-not (Archive-NonResumableFailedArtifact)) { throw ('V25e artifact has no resumable last_model.pt; preserve it for inspection before retrying: ' + $artifactPath) }
        }
    }
    Assert-GpuMemoryHeadroom
    Invoke-RadarPython $arguments
    Assert-CompletedTraining
}

function Invoke-Evaluate {
    Assert-CompletedTraining
    Invoke-RadarPython @('-m', 'report.known_gate_v25e', '--project-root', $projectPath)
    Assert-File $reportPath 'V25e paired development report'
    $report = Read-Utf8Json $reportPath
    if ([string]$report.status -ne 'completed_development_paired_gradient_projection_experiment' -or [bool]$report.deployment_eligible) { throw ('V25e report violates sealed development scope: ' + $reportPath) }
}

try {
    Assert-File $python 'Project Python interpreter'
    Write-Host ('[V25e] project=' + $projectPath) -ForegroundColor Green
    Write-Host ('[V25e] stage=' + $Stage + '; V25a control versus shared-gradient conflict projection treatment; full20epoch') -ForegroundColor Green
    Write-Host '[V25e] OOF, test, Unknown, L3, full cascade, rejection, stable and P95 remain intentionally unexecuted.' -ForegroundColor Green
    switch ($Stage) {
        'prepare' { Invoke-Prepare }
        'smoke' { Invoke-Smoke }
        'train' { Invoke-Train }
        'evaluate' { Invoke-Evaluate }
        'all' { Invoke-Prepare; Invoke-Smoke; Invoke-Train; Invoke-Evaluate }
    }
    Write-Host '[V25e] requested stage completed.' -ForegroundColor Green
}
catch { [Console]::Error.WriteLine('[V25e] pipeline stopped: ' + $_.Exception.Message); exit 1 }
