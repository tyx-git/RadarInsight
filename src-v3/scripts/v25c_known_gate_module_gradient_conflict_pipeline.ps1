[CmdletBinding()]
param(
    [ValidateSet('prepare', 'smoke', 'analyze', 'evaluate', 'all')]
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
$readinessPath = Join-Path $projectPath 'src-v3\processed\report\RADAR-LN3-V25c-known-gate-exact-replay-module-gradient-conflict-localization-readiness.json'
$contractPath = Join-Path $projectPath 'data\processed\RADAR-LN3-V25c-cq08-known-gate-exact-replay-module-gradient-conflict-localization-contract.json'
$reportPath = Join-Path $projectPath 'src-v3\processed\report\RADAR-LN3-V25c-known-gate-exact-replay-module-gradient-conflict-localization.json'
$completionPath = Join-Path $projectPath 'src-v3\processed\report\RADAR-LN3-V25c-known-gate-exact-replay-module-gradient-conflict-localization-completion.json'
$sourceArtifactPath = Join-Path $projectPath 'artifacts\RADAR-LN3-V25a-cq08-liquid-known-gate-course-preclip-gradient-telemetry-seed43-formal'
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

function Assert-SourceArtifact {
    Assert-File (Join-Path $sourceArtifactPath 'training-summary.json') 'Completed V25a training summary'
    Assert-File (Join-Path $sourceArtifactPath 'gradient-telemetry\metadata.json') 'V25a telemetry metadata'
    for ($epoch = 1; $epoch -le 20; $epoch++) {
        Assert-File (Join-Path $sourceArtifactPath ('epochs\epoch-' + $epoch.ToString('000') + '.pt')) ('V25a epoch checkpoint ' + $epoch)
        Assert-File (Join-Path $sourceArtifactPath ('gradient-telemetry\epoch-' + $epoch.ToString('000') + '.json')) ('V25a epoch telemetry ' + $epoch)
    }
    $summary = Read-Utf8Json (Join-Path $sourceArtifactPath 'training-summary.json')
    if ([string]$summary.status -ne 'completed' -or [string]$summary.variant -ne 'v25a_course_preclip_gradient_telemetry' -or [int]$summary.global_step -ne 9860 -or $summary.deployment_eligible -ne $false) {
        throw 'V25a source artifact is incomplete or outside V25c sealed replay scope.'
    }
}

function Assert-Readiness {
    Assert-File $readinessPath 'V25c readiness'
    Assert-File $contractPath 'V25c experiment contract'
    $readiness = Read-Utf8Json $readinessPath
    $contract = Read-Utf8Json $contractPath
    $issues = [System.Collections.Generic.List[string]]::new()
    if ([string]$readiness.status -ne 'passed') { $issues.Add('status') }
    if ([string]$readiness.schema_version -ne 'RADAR-LN3-V25c-known-gate-exact-replay-module-gradient-conflict-localization-readiness-v1') { $issues.Add('schema_version') }
    if ([string]$readiness.variant -ne 'v25c_v25a_exact_replay_module_gradient_conflict_localization') { $issues.Add('variant') }
    $expectedContractPath = 'data/processed/RADAR-LN3-V25c-cq08-known-gate-exact-replay-module-gradient-conflict-localization-contract.json'
    $contractHash = (Get-FileHash -LiteralPath $contractPath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ([string]$readiness.experiment_contract.path -ne $expectedContractPath -or [string]$readiness.experiment_contract.sha256 -ne $contractHash) { $issues.Add('contract_binding') }
    if ([string]$contract.schema_version -ne 'RADAR-LN3-V25c-known-gate-exact-replay-module-gradient-conflict-localization-experiment-contract-v1' -or [string]$contract.internal_variant -ne 'v25c_v25a_exact_replay_module_gradient_conflict_localization' -or $contract.replay_verification.prepared_source_hash_binding_must_match_exactly_before_replay -ne $true) { $issues.Add('contract_identity') }
    if ($null -eq $contract.implementation_sha256 -or @($contract.implementation_sha256.PSObject.Properties).Count -lt 25) { $issues.Add('implementation_binding') }
    if ([int]$readiness.expected_total_steps -ne 9860 -or [int]$readiness.expected_steps_per_epoch -ne 493) { $issues.Add('replay_steps') }
    if ([bool]$readiness.validation_used_for_readiness_gate) { $issues.Add('validation_gate') }
    if ([int]$readiness.validation_payload_access_count -ne 0 -or [int]$readiness.test_payload_access_count -ne 0 -or [int]$readiness.unknown_payload_access_count -ne 0) { $issues.Add('sealed_payload_scope') }
    if ([bool]$readiness.deployment_eligible) { $issues.Add('deployment') }
    if ($readiness.checks.module_partition_must_cover_every_trainable_parameter_once -ne $true -or $readiness.checks.final_and_prefix_component_additivity_must_be_checked -ne $true) { $issues.Add('module_component_contract') }
    if ($readiness.checks.bounded_worker_sidecar_and_compact_completion_are_required -ne $true -or $contract.replay_verification.bounded_sidecar_transport_and_compact_completion_are_required -ne $true) { $issues.Add('bounded_transport_contract') }
    if ($contract.execution_resource_safety_policy.epoch_worker_transport -ne 'atomic_json_sidecar_in_project_tmp_not_stdout' -or [int]$contract.execution_resource_safety_policy.worker_payload_max_bytes -ne 16777216 -or $contract.execution_resource_safety_policy.completed_report_stdout -ne 'compact_completion_only') { $issues.Add('resource_policy') }
    foreach ($property in @($readiness.checks.PSObject.Properties)) {
        if ([bool]$property.Value -ne $true) { $issues.Add('check:' + $property.Name) }
    }
    if ($issues.Count -ne 0) { throw ('V25c readiness is stale or failed: ' + ($issues -join ',') + '; path=' + $readinessPath) }
}

function Archive-StalePreparationIfSafe {
    # Preparation artifacts bind the current implementation byte-for-byte.
    # When V25c code changes before any report exists, retain the old binding
    # under tmp and regenerate instead of asking the user to delete a frozen
    # artifact manually. A completed report is evidence and is never moved.
    if (Test-Path -LiteralPath $reportPath -PathType Leaf) { return }
    $existing = @($contractPath, $readinessPath) | Where-Object { Test-Path -LiteralPath $_ -PathType Leaf }
    if ($existing.Count -eq 0) { return }
    Push-Location $srcRoot
    try {
        $previousErrorActionPreference = $ErrorActionPreference
        $ErrorActionPreference = 'Continue'
        $bindingResult = & $python -X utf8 -m build.known_gate_v25c --project-root $projectPath --check-prepared-binding 2>&1
        $bindingExitCode = [int]$LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousErrorActionPreference
        Pop-Location
    }
    if ($bindingExitCode -eq 0) {
        Write-Host '[V25c] existing preparation remains byte-bound to the current implementation.' -ForegroundColor DarkGray
        return
    }
    $stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
    $archive = Join-Path $projectPath ('tmp\v25c-pre-prepare-regeneration-' + $stamp)
    New-Item -ItemType Directory -Path $archive -Force | Out-Null
    foreach ($path in $existing) {
        Move-Item -LiteralPath $path -Destination (Join-Path $archive ([System.IO.Path]::GetFileName($path)))
    }
    if (Test-Path -LiteralPath $completionPath -PathType Leaf) {
        Move-Item -LiteralPath $completionPath -Destination (Join-Path $archive ([System.IO.Path]::GetFileName($completionPath)))
    }
    Write-Host ('[V25c] archived stale pre-analysis preparation to ' + $archive) -ForegroundColor Yellow
}

function Assert-GpuMemoryHeadroom {
    $nvidiaSmi = Get-Command 'nvidia-smi.exe' -ErrorAction SilentlyContinue
    if ($null -eq $nvidiaSmi) { throw 'nvidia-smi.exe is unavailable; V25c exact replay requires a CUDA headroom audit.' }
    $output = @(& $nvidiaSmi.Source @('--id=0', '--query-gpu=memory.free,memory.total', '--format=csv,noheader,nounits'))
    if ([int]$LASTEXITCODE -ne 0 -or $output.Count -ne 1) { throw 'Unable to query GPU 0 memory with nvidia-smi.' }
    $freeMiB = [int]::Parse((([string]$output[0] -split ',')[0]).Trim(), [System.Globalization.CultureInfo]::InvariantCulture)
    Write-Host ('[V25c] GPU0 free=' + $freeMiB + ' MiB; required=' + $minimumFreeGpuMemoryMiB + ' MiB') -ForegroundColor Cyan
    if ($freeMiB -lt $minimumFreeGpuMemoryMiB) { throw ('V25c CUDA preflight blocked: free GPU memory is below ' + $minimumFreeGpuMemoryMiB + ' MiB.') }
}

function Assert-CompletedAnalysis {
    Assert-File $reportPath 'V25c exact replay report'
    Assert-File $completionPath 'V25c exact replay completion record'
    $completion = Read-Utf8Json $completionPath
    if ([string]$completion.status -ne 'completed_development_exact_replay_module_gradient_conflict_diagnosis_only' -or [string]$completion.schema_version -ne 'RADAR-LN3-V25c-known-gate-exact-replay-module-gradient-conflict-localization-completion-v1') { throw 'V25c completion record has an invalid identity.' }
    if ([string]$completion.report_path -ne $reportPath) { throw 'V25c completion record points to a different report.' }
    $reportHash = (Get-FileHash -LiteralPath $reportPath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ([string]$completion.report_sha256 -ne $reportHash -or [int64]$completion.report_bytes -ne [int64](Get-Item -LiteralPath $reportPath).Length) { throw 'V25c completion record does not bind the exact report bytes.' }
    if ([int]$completion.replayed_step_count -ne 9860 -or [int]$completion.expected_step_count -ne 9860 -or [int]$completion.replayed_epoch_count -ne 20 -or $completion.every_parent_batch_hash_matched_exactly -ne $true -or $completion.every_clip_decision_matched_exactly -ne $true) { throw 'V25c completion record does not prove complete exact replay.' }
    if ([int]$completion.mixed_known_clutter_batch_count -ne 638 -or [int]$completion.expected_mixed_known_clutter_batch_count -ne 638) { throw 'V25c completion record has incomplete mixed Known/Clutter support.' }
    if ([int]$completion.validation_payload_access_count -ne 0 -or [int]$completion.test_payload_access_count -ne 0 -or [int]$completion.unknown_payload_access_count -ne 0 -or [bool]$completion.checkpoint_written -or [bool]$completion.optimizer_state_written -or [bool]$completion.checkpoint_selection_performed) { throw 'V25c completion record violates diagnostic-only sealed scope.' }
    if ([bool]$completion.deployment_eligible) { throw 'V25c completion record must not be deployment eligible.' }
    if ($completion.execution_resource_safety_policy.epoch_worker_transport -ne 'atomic_json_sidecar_in_project_tmp_not_stdout' -or [int]$completion.execution_resource_safety_policy.worker_payload_max_bytes -ne 16777216 -or $completion.execution_resource_safety_policy.completed_report_stdout -ne 'compact_completion_only') { throw 'V25c completion record lacks the required bounded resource policy.' }
}

function Invoke-Prepare {
    Invoke-RadarPython @('-m', 'py_compile', 'radar_ln3\known_gate_v25c_module_gradient_conflict.py', 'build\known_gate_v25c.py', 'report\known_gate_v25c.py', 'smoke\known_gate_v25c_contract_test.py')
    Archive-StalePreparationIfSafe
    Invoke-RadarPython @('-m', 'build.known_gate_v25c', '--project-root', $projectPath)
    Assert-Readiness
}

function Invoke-Smoke {
    Assert-SourceArtifact
    Assert-Readiness
    Assert-GpuMemoryHeadroom
    Invoke-RadarPython @('-m', 'smoke.known_gate_v25c_contract_test', '--project-root', $projectPath)
    Invoke-RadarPython @('-m', 'smoke.static_isolation_check', '--root', $srcRoot)
}

function Invoke-Analyze {
    Assert-SourceArtifact
    Assert-Readiness
    Assert-GpuMemoryHeadroom
    Invoke-RadarPython @('-m', 'report.known_gate_v25c', '--project-root', $projectPath)
    Assert-CompletedAnalysis
}

function Invoke-Evaluate {
    Assert-Readiness
    Assert-CompletedAnalysis
}

try {
    Assert-File $python 'Project Python interpreter'
    Write-Host ('[V25c] project=' + $projectPath) -ForegroundColor Green
    Write-Host ('[V25c] stage=' + $Stage + '; V25a exact replay module-level mixed Known/Clutter gradient conflict localization') -ForegroundColor Green
    Write-Host '[V25c] Read-only diagnosis: no training, checkpoint mutation, validation/test/Unknown/L3/cascade/stable/P95 evaluation, or deployment claim.' -ForegroundColor Green
    switch ($Stage) {
        'prepare' { Invoke-Prepare }
        'smoke' { Invoke-Smoke }
        'analyze' { Invoke-Analyze }
        'evaluate' { Invoke-Evaluate }
        'all' { Invoke-Prepare; Invoke-Smoke; Invoke-Analyze; Invoke-Evaluate }
    }
    Write-Host '[V25c] requested stage completed.' -ForegroundColor Green
}
catch { [Console]::Error.WriteLine('[V25c] pipeline stopped: ' + $_.Exception.Message); exit 1 }
