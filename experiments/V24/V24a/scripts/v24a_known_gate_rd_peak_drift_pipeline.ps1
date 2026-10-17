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
$artifactPath = Join-Path $projectPath 'artifacts\RADAR-LN3-V24a-cq08-liquid-known-gate-course-rd-peak-drift-mean-seed43-formal'
$readinessPath = Join-Path $projectPath 'src-v3\processed\report\RADAR-LN3-V24a-known-gate-course-rd-peak-drift-mean-training-readiness.json'
$reportPath = Join-Path $projectPath 'src-v3\processed\report\RADAR-LN3-V24a-known-gate-course-rd-peak-drift-mean-paired-development.json'
$minimumFreeGpuMemoryMiB = 4096

function Invoke-RadarPython {
    param([Parameter(Mandatory = $true)][string[]]$Arguments)
    $runtimeArguments = @('-X', 'utf8') + $Arguments
    $display = $python + ' ' + ($runtimeArguments -join ' ')
    Write-Host ('> ' + $display) -ForegroundColor Cyan
    $exitCode = 0
    Push-Location $srcRoot
    try {
        & $python @runtimeArguments | Out-Host
        $exitCode = [int]$LASTEXITCODE
    }
    finally { Pop-Location }
    if ($exitCode -ne 0) { throw ('Python command failed with exit code ' + $exitCode + ': ' + $display) }
}

function Assert-File {
    param([Parameter(Mandatory = $true)][string]$Path, [Parameter(Mandatory = $true)][string]$Description)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { throw ($Description + ' is missing: ' + $Path) }
}

function Read-Utf8Json {
    param([Parameter(Mandatory = $true)][string]$Path)
    return ConvertFrom-Json -InputObject ([System.IO.File]::ReadAllText($Path, [System.Text.UTF8Encoding]::new($false, $true)))
}

function Assert-Readiness {
    Assert-File $readinessPath 'V24a readiness'
    $readiness = Read-Utf8Json $readinessPath
    $issues = [System.Collections.Generic.List[string]]::new()
    if ([string]$readiness.status -ne 'passed') { $issues.Add('status') }
    if ([string]$readiness.schema_version -ne 'RADAR-LN3-V24a-known-gate-course-rd-peak-drift-mean-training-readiness-v1') { $issues.Add('schema_version') }
    if ([string]$readiness.variant -ne 'v24a_course_rd_peak_drift_mean') { $issues.Add('variant') }
    if ([int]$readiness.known_train_parent_count -ne 986 -or [int]$readiness.known_validation_parent_count -ne 232 -or [int]$readiness.validation_clutter_parent_support -ne 9) { $issues.Add('support') }
    if ([int]$readiness.unknown_training_exposure -ne 0) { $issues.Add('unknown_training_exposure') }
    if ([bool]$readiness.validation_used_for_readiness_gate) { $issues.Add('validation_used_for_readiness_gate') }
    if ([int]$readiness.test_payload_access_count -ne 0 -or [bool]$readiness.deployment_eligible) { $issues.Add('sealed_scope') }
    foreach ($property in @($readiness.checks.PSObject.Properties)) {
        if ([bool]$property.Value -ne $true) { $issues.Add('check:' + $property.Name) }
    }
    if ($issues.Count -ne 0) { throw ('V24a readiness is stale or failed: ' + ($issues -join ',') + '; path=' + $readinessPath) }
}

function Assert-GpuMemoryHeadroom {
    $nvidiaSmi = Get-Command 'nvidia-smi.exe' -ErrorAction SilentlyContinue
    if ($null -eq $nvidiaSmi) { throw 'nvidia-smi.exe is unavailable; V24a training requires a CUDA headroom audit.' }
    $output = @(& $nvidiaSmi.Source @('--id=0', '--query-gpu=memory.free,memory.total', '--format=csv,noheader,nounits'))
    if ([int]$LASTEXITCODE -ne 0 -or $output.Count -ne 1) { throw 'Unable to query GPU 0 memory with nvidia-smi.' }
    $freeMiB = [int]::Parse((([string]$output[0] -split ',')[0]).Trim(), [System.Globalization.CultureInfo]::InvariantCulture)
    Write-Host ('[V24a] GPU0 free=' + $freeMiB + ' MiB; required=' + $minimumFreeGpuMemoryMiB + ' MiB') -ForegroundColor Cyan
    if ($freeMiB -lt $minimumFreeGpuMemoryMiB) { throw ('V24a CUDA preflight blocked: free GPU memory is below ' + $minimumFreeGpuMemoryMiB + ' MiB.') }
}

function Assert-CompletedTreatment {
    $summaryPath = Join-Path $artifactPath 'training-summary.json'
    Assert-File $summaryPath 'V24a training summary'
    Assert-File (Join-Path $artifactPath 'best_model.pt') 'V24a best checkpoint'
    Assert-File (Join-Path $artifactPath 'last_model.pt') 'V24a last checkpoint'
    Assert-File (Join-Path $artifactPath 'sample_detail_information.json') 'V24a sample detail'
    for ($epoch = 1; $epoch -le 20; $epoch++) {
        Assert-File (Join-Path $artifactPath ('epochs\epoch-' + $epoch.ToString('000') + '.pt')) ('V24a epoch ' + $epoch)
    }
    $summary = Read-Utf8Json $summaryPath
    if ([string]$summary.status -ne 'completed' -or [string]$summary.variant -ne 'v24a_course_rd_peak_drift_mean') { throw ('V24a completed artifact identity is invalid: ' + $summaryPath) }
    if (@($summary.history).Count -ne 20 -or $summary.full_parameter_training -ne $true) { throw 'V24a full 20-epoch training contract is invalid.' }
    if ($summary.paired_initialization.sealed_v24_control_initial_model_state_reproduced -ne $true) { throw 'V24a V24 initialization audit is missing from the completed summary.' }
    if ($summary.paired_initialization.real_training_probe_rd_drift_column_receives_gradient -ne $true) { throw 'V24a RD drift gradient audit is missing from the completed summary.' }
    if ($summary.test_evaluation -ne 'not_executed' -or $summary.deployment_eligible -ne $false) { throw 'V24a completed scope is not sealed.' }
}

function Invoke-Prepare {
    Invoke-RadarPython @('-m', 'py_compile', 'radar_ln3\known_gate_features.py', 'radar_ln3\known_gate.py', 'build\known_gate_v24a.py', 'train\known_gate_common.py', 'train\known_gate_v24a.py', 'report\known_gate_v24a.py')
    Invoke-RadarPython @('-m', 'build.known_gate_v24a', '--project-root', $projectPath)
    Assert-Readiness
}


function Archive-NonResumableFailedArtifact {
    if (-not (Test-Path -LiteralPath $artifactPath -PathType Container)) { return $false }
    $summaryPath = Join-Path $artifactPath 'training-summary.json'
    $lastCheckpoint = Join-Path $artifactPath 'last_model.pt'
    if (Test-Path -LiteralPath $lastCheckpoint -PathType Leaf) { return $false }
    $archiveReason = ''
    if (Test-Path -LiteralPath $summaryPath -PathType Leaf) {
        $summary = Read-Utf8Json $summaryPath
        if ([string]$summary.status -ne 'failed') { return $false }
        $archiveReason = 'failed-before-checkpoint'
    }
    else {
        $items = @(Get-ChildItem -LiteralPath $artifactPath -Force)
        $nonLogItems = @($items | Where-Object { $_.Name -ne 'training.log' })
        if ($items.Count -eq 0 -or $nonLogItems.Count -ne 0 -or -not (Test-Path -LiteralPath (Join-Path $artifactPath 'training.log') -PathType Leaf)) { return $false }
        $archiveReason = 'interrupted-before-checkpoint'
    }
    $timestamp = (Get-Date).ToString('yyyyMMdd-HHmmss-fff', [System.Globalization.CultureInfo]::InvariantCulture)
    $archivePath = $artifactPath + '.' + $archiveReason + '-' + $timestamp
    if (Test-Path -LiteralPath $archivePath) { throw ('V24a failed-artifact archive path already exists: ' + $archivePath) }
    Move-Item -LiteralPath $artifactPath -Destination $archivePath
    Write-Host ('[V24a] archived non-resumable ' + $archiveReason + ' attempt to ' + $archivePath) -ForegroundColor Yellow
    return $true
}

function Invoke-Train {
    Assert-Readiness
    $summaryPath = Join-Path $artifactPath 'training-summary.json'
    $lastCheckpoint = Join-Path $artifactPath 'last_model.pt'
    if (Test-Path -LiteralPath $summaryPath -PathType Leaf) {
        $summary = Read-Utf8Json $summaryPath
        if ([string]$summary.status -eq 'completed') { Assert-CompletedTreatment; Write-Host '[V24a] completed treatment exists; preserving it.' -ForegroundColor Yellow; return }
    }
    $arguments = @('-m', 'train.known_gate_v24a', '--project-root', $projectPath)
    if (Test-Path -LiteralPath $artifactPath -PathType Container) {
        $items = @(Get-ChildItem -LiteralPath $artifactPath -Force)
        if ($items.Count -gt 0) {
            if (Test-Path -LiteralPath $lastCheckpoint -PathType Leaf) { $arguments += @('--resume-checkpoint', $lastCheckpoint); Write-Host ('[V24a] resuming from ' + $lastCheckpoint) -ForegroundColor Yellow }
            elseif (-not (Archive-NonResumableFailedArtifact)) { throw ('V24a artifact has no resumable last_model.pt; preserve it for inspection before retrying: ' + $artifactPath) }
        }
    }
    Assert-GpuMemoryHeadroom
    Invoke-RadarPython $arguments
    Assert-CompletedTreatment
}

function Invoke-Evaluate {
    Assert-CompletedTreatment
    Invoke-RadarPython @('-m', 'report.known_gate_v24a', '--project-root', $projectPath)
    Assert-File $reportPath 'V24a paired development report'
    $report = Read-Utf8Json $reportPath
    if ([string]$report.status -ne 'completed_development_paired_only' -or $report.deployment_eligible -ne $false) { throw ('V24a report violates sealed scope: ' + $reportPath) }
    if ($null -eq $report.course_incoherence_strata_by_scope -or $null -eq $report.rd_drift_validity_strata_by_scope -or $null -eq $report.error_direction_audit.high_course_bird_by_scope -or $null -eq $report.capture_parent_strata -or $null -eq $report.per_parent_changes) { throw 'V24a report lacks required scope strata or parent-change audit.' }
    foreach ($scope in @('parent_final', 'eligible_prefix', 'window')) {
        if ($null -eq $report.course_incoherence_strata_by_scope.$scope -or $null -eq $report.rd_drift_validity_strata_by_scope.$scope -or $null -eq $report.error_direction_audit.high_course_bird_by_scope.$scope) { throw ('V24a report lacks required ' + $scope + ' audit.') }
    }
}

try {
    Assert-File $python 'Project Python interpreter'
    Write-Host ('[V24a] project=' + $projectPath) -ForegroundColor Green
    Write-Host ('[V24a] stage=' + $Stage + '; V24 plus exactly one current-event RD peak-drift feature, full20epoch') -ForegroundColor Green
    Write-Host '[V24a] A current event with fewer than two RD frames is invalid, not a valid zero; no preceding-event bridge is used. OOF, test, Unknown, L3, full cascade, rejection, stable and P95 remain not_executed.' -ForegroundColor Green
    switch ($Stage) {
        'prepare' { Invoke-Prepare }
        'train' { Invoke-Train }
        'evaluate' { Invoke-Evaluate }
        'all' { Invoke-Prepare; Invoke-Train; Invoke-Evaluate }
    }
    Write-Host '[V24a] requested stage completed.' -ForegroundColor Green
}
catch { [Console]::Error.WriteLine('[V24a] pipeline stopped: ' + $_.Exception.Message); exit 1 }
