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
$artifactPath = Join-Path $projectPath 'artifacts\RADAR-LN3-V24c-cq08-liquid-known-gate-course-strict-same-second-skip-seed43-formal'
$readinessPath = Join-Path $projectPath 'src-v3\processed\report\RADAR-LN3-V24c-known-gate-course-strict-same-second-skip-training-readiness.json'
$reportPath = Join-Path $projectPath 'src-v3\processed\report\RADAR-LN3-V24c-known-gate-course-strict-same-second-skip-paired-development.json'
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
    if ($exitCode -ne 0) {
        throw ('Python command failed with exit code ' + $exitCode + ': ' + $display)
    }
}

function Assert-File {
    param([Parameter(Mandatory = $true)][string]$Path, [Parameter(Mandatory = $true)][string]$Description)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw ($Description + ' is missing: ' + $Path)
    }
}

function Read-Utf8Json {
    param([Parameter(Mandatory = $true)][string]$Path)
    return ConvertFrom-Json -InputObject ([System.IO.File]::ReadAllText($Path, [System.Text.UTF8Encoding]::new($false, $true)))
}

function Assert-Readiness {
    Assert-File $readinessPath 'V24c readiness'
    $readiness = Read-Utf8Json $readinessPath
    $issues = [System.Collections.Generic.List[string]]::new()
    if ([string]$readiness.status -ne 'passed') { $issues.Add('status') }
    if ([string]$readiness.schema_version -ne 'RADAR-LN3-V24c-known-gate-course-strict-same-second-skip-training-readiness-v1') { $issues.Add('schema_version') }
    if ([string]$readiness.variant -ne 'v24c_course_strict_same_second_skip') { $issues.Add('variant') }
    if ([int]$readiness.known_train_parent_count -ne 986 -or [int]$readiness.known_validation_parent_count -ne 232 -or [int]$readiness.validation_clutter_parent_support -ne 9) { $issues.Add('support') }
    if ([int]$readiness.unknown_training_exposure -ne 0) { $issues.Add('unknown_training_exposure') }
    if ([bool]$readiness.validation_used_for_readiness_gate) { $issues.Add('validation_used_for_readiness_gate') }
    if ([int]$readiness.test_payload_access_count -ne 0 -or [bool]$readiness.deployment_eligible) { $issues.Add('sealed_scope') }
    if ([string]$readiness.formal_training -ne 'not_executed' -and -not (Test-Path -LiteralPath (Join-Path $artifactPath 'training-summary.json') -PathType Leaf)) { $issues.Add('formal_training_state') }
    if ($null -eq $readiness.initialization -or $readiness.initialization.only_dynamics_mode_differs -ne $true -or $readiness.initialization.control_mode -ne 'bounded_observation' -or $readiness.initialization.treatment_mode -ne 'strict_skip') { $issues.Add('dynamics_only_initialization') }
    if ($null -eq $readiness.same_second_support_diagnostic -or [int]$readiness.same_second_support_diagnostic.train.by_base_label.ClutterTarget.parents_with_same_second_followup -ne 0 -or [int]$readiness.same_second_support_diagnostic.val.by_base_label.ClutterTarget.parents_with_same_second_followup -ne 0) { $issues.Add('same_second_support') }
    foreach ($property in @($readiness.checks.PSObject.Properties)) {
        if ([bool]$property.Value -ne $true) { $issues.Add('check:' + $property.Name) }
    }
    if ($issues.Count -ne 0) {
        throw ('V24c readiness is stale or failed: ' + ($issues -join ',') + '; path=' + $readinessPath)
    }
}

function Assert-GpuMemoryHeadroom {
    $nvidiaSmi = Get-Command 'nvidia-smi.exe' -ErrorAction SilentlyContinue
    if ($null -eq $nvidiaSmi) { throw 'nvidia-smi.exe is unavailable; V24c training requires a CUDA headroom audit.' }
    $output = @(& $nvidiaSmi.Source @('--id=0', '--query-gpu=memory.free,memory.total', '--format=csv,noheader,nounits'))
    if ([int]$LASTEXITCODE -ne 0 -or $output.Count -ne 1) { throw 'Unable to query GPU 0 memory with nvidia-smi.' }
    $freeMiB = [int]::Parse((([string]$output[0] -split ',')[0]).Trim(), [System.Globalization.CultureInfo]::InvariantCulture)
    Write-Host ('[V24c] GPU0 free=' + $freeMiB + ' MiB; required=' + $minimumFreeGpuMemoryMiB + ' MiB') -ForegroundColor Cyan
    if ($freeMiB -lt $minimumFreeGpuMemoryMiB) { throw ('V24c CUDA preflight blocked: free GPU memory is below ' + $minimumFreeGpuMemoryMiB + ' MiB.') }
}

function Assert-CompletedTreatment {
    $summaryPath = Join-Path $artifactPath 'training-summary.json'
    Assert-File $summaryPath 'V24c training summary'
    Assert-File (Join-Path $artifactPath 'best_model.pt') 'V24c best checkpoint'
    Assert-File (Join-Path $artifactPath 'last_model.pt') 'V24c last checkpoint'
    Assert-File (Join-Path $artifactPath 'sample_detail_information.json') 'V24c sample detail'
    for ($epoch = 1; $epoch -le 20; $epoch++) {
        Assert-File (Join-Path $artifactPath ('epochs\epoch-' + $epoch.ToString('000') + '.pt')) ('V24c epoch ' + $epoch)
    }
    $summary = Read-Utf8Json $summaryPath
    if ([string]$summary.status -ne 'completed' -or [string]$summary.variant -ne 'v24c_course_strict_same_second_skip') { throw ('V24c completed artifact identity is invalid: ' + $summaryPath) }
    if (@($summary.history).Count -ne 20 -or $summary.full_parameter_training -ne $true) { throw 'V24c full 20-epoch training contract is invalid.' }
    if ($summary.paired_initialization.sealed_v24_control_initial_model_state_reproduced -ne $true -or $summary.paired_initialization.initial_logits_exactly_equal_without_zero_delta -ne $true -or $summary.paired_initialization.strict_skip_state_unchanged_at_every_probe_same_second_event -ne $true) { throw 'V24c strict-skip initialization audit is missing from the completed summary.' }
    if ($summary.test_evaluation -ne 'not_executed' -or $summary.deployment_eligible -ne $false) { throw 'V24c completed scope is not sealed.' }
}

function Invoke-Prepare {
    Invoke-RadarPython @('-m', 'py_compile', 'radar_ln3\model.py', 'radar_ln3\config.py', 'radar_ln3\task_models.py', 'radar_ln3\known_gate_features.py', 'radar_ln3\known_gate.py', 'build\known_gate_v24c.py', 'train\known_gate_common.py', 'train\known_gate_v24c.py', 'report\known_gate_v24c.py')
    Invoke-RadarPython @('-m', 'build.known_gate_v24c', '--project-root', $projectPath)
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
    if (Test-Path -LiteralPath $archivePath) { throw ('V24c failed-artifact archive path already exists: ' + $archivePath) }
    Move-Item -LiteralPath $artifactPath -Destination $archivePath
    Write-Host ('[V24c] archived non-resumable ' + $archiveReason + ' attempt to ' + $archivePath) -ForegroundColor Yellow
    return $true
}

function Invoke-Train {
    Assert-Readiness
    $summaryPath = Join-Path $artifactPath 'training-summary.json'
    $lastCheckpoint = Join-Path $artifactPath 'last_model.pt'
    if (Test-Path -LiteralPath $summaryPath -PathType Leaf) {
        $summary = Read-Utf8Json $summaryPath
        if ([string]$summary.status -eq 'completed') { Assert-CompletedTreatment; Write-Host '[V24c] completed treatment exists; preserving it.' -ForegroundColor Yellow; return }
    }
    $arguments = @('-m', 'train.known_gate_v24c', '--project-root', $projectPath)
    if (Test-Path -LiteralPath $artifactPath -PathType Container) {
        $items = @(Get-ChildItem -LiteralPath $artifactPath -Force)
        if ($items.Count -gt 0) {
            if (Test-Path -LiteralPath $lastCheckpoint -PathType Leaf) { $arguments += @('--resume-checkpoint', $lastCheckpoint); Write-Host ('[V24c] resuming from ' + $lastCheckpoint) -ForegroundColor Yellow }
            elseif (-not (Archive-NonResumableFailedArtifact)) { throw ('V24c artifact has no resumable last_model.pt; preserve it for inspection before retrying: ' + $artifactPath) }
        }
    }
    Assert-GpuMemoryHeadroom
    Invoke-RadarPython $arguments
    Assert-CompletedTreatment
}

function Invoke-Evaluate {
    Assert-CompletedTreatment
    Invoke-RadarPython @('-m', 'report.known_gate_v24c', '--project-root', $projectPath)
    Assert-File $reportPath 'V24c paired development report'
    $report = Read-Utf8Json $reportPath
    if ([string]$report.status -ne 'completed_development_paired_only' -or $report.deployment_eligible -ne $false) { throw ('V24c report violates sealed scope: ' + $reportPath) }
    if ($null -eq $report.same_second_reason_strata_by_scope -or $null -eq $report.same_second_ordinal_strata_by_scope -or $null -eq $report.same_second_parent_followup_strata -or $null -eq $report.same_second_parent_multiplicity_strata -or $null -eq $report.track_length_parent_strata -or $null -eq $report.capture_parent_strata -or $null -eq $report.per_parent_changes) { throw 'V24c report lacks required dynamics strata or parent-change audit.' }
}

try {
    Assert-File $python 'Project Python interpreter'
    Write-Host ('[V24c] project=' + $projectPath) -ForegroundColor Green
    Write-Host ('[V24c] stage=' + $Stage + '; V24 strict same-second recurrent-state skip treatment, full20epoch') -ForegroundColor Green
    Write-Host '[V24c] The V24 course cache is reused byte-for-byte. V24 trained weights are not an initialization source. OOF, test, Unknown, L3, full cascade, rejection, stable and P95 remain not_executed.' -ForegroundColor Green
    switch ($Stage) {
        'prepare' { Invoke-Prepare }
        'train' { Invoke-Train }
        'evaluate' { Invoke-Evaluate }
        'all' { Invoke-Prepare; Invoke-Train; Invoke-Evaluate }
    }
    Write-Host '[V24c] requested stage completed.' -ForegroundColor Green
}
catch { [Console]::Error.WriteLine('[V24c] pipeline stopped: ' + $_.Exception.Message); exit 1 }
