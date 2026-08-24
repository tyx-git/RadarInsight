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
$readinessPath = Join-Path $projectPath 'src-v3\processed\report\RADAR-LN3-V25b-known-gate-exact-replay-parent-gradient-decomposition-readiness.json'
$contractPath = Join-Path $projectPath 'data\processed\RADAR-LN3-V25b-cq08-known-gate-exact-replay-parent-gradient-decomposition-contract.json'
$reportPath = Join-Path $projectPath 'src-v3\processed\report\RADAR-LN3-V25b-known-gate-exact-replay-parent-gradient-decomposition.json'
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
        throw 'V25a source artifact is incomplete or outside V25b sealed replay scope.'
    }
}

function Assert-Readiness {
    Assert-File $readinessPath 'V25b readiness'
    Assert-File $contractPath 'V25b experiment contract'
    $readiness = Read-Utf8Json $readinessPath
    $contract = Read-Utf8Json $contractPath
    $issues = [System.Collections.Generic.List[string]]::new()
    if ([string]$readiness.status -ne 'passed') { $issues.Add('status') }
    if ([string]$readiness.schema_version -ne 'RADAR-LN3-V25b-known-gate-exact-replay-parent-gradient-decomposition-readiness-v1') { $issues.Add('schema_version') }
    if ([string]$readiness.variant -ne 'v25b_v25a_exact_replay_parent_gradient_decomposition') { $issues.Add('variant') }
    $expectedContractPath = 'data/processed/RADAR-LN3-V25b-cq08-known-gate-exact-replay-parent-gradient-decomposition-contract.json'
    $contractHash = (Get-FileHash -LiteralPath $contractPath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ([string]$readiness.experiment_contract.path -ne $expectedContractPath -or [string]$readiness.experiment_contract.sha256 -ne $contractHash) { $issues.Add('contract_binding') }
    if ([string]$contract.schema_version -ne 'RADAR-LN3-V25b-known-gate-exact-replay-parent-gradient-decomposition-experiment-contract-v1' -or [string]$contract.internal_variant -ne 'v25b_v25a_exact_replay_parent_gradient_decomposition' -or $contract.replay_verification.prepared_source_hash_binding_must_match_exactly_before_replay -ne $true) { $issues.Add('contract_identity') }
    if ($null -eq $contract.implementation_sha256 -or @($contract.implementation_sha256.PSObject.Properties).Count -ne 19) { $issues.Add('implementation_binding') }
    if ([int]$readiness.expected_total_steps -ne 9860 -or [int]$readiness.expected_steps_per_epoch -ne 493) { $issues.Add('replay_steps') }
    if ([bool]$readiness.validation_used_for_readiness_gate) { $issues.Add('validation_gate') }
    if ([int]$readiness.validation_payload_access_count -ne 0 -or [int]$readiness.test_payload_access_count -ne 0 -or [int]$readiness.unknown_payload_access_count -ne 0) { $issues.Add('sealed_payload_scope') }
    if ([bool]$readiness.deployment_eligible) { $issues.Add('deployment') }
    if ($readiness.checks.prepared_source_hash_binding_is_rechecked_before_replay -ne $true) { $issues.Add('source_hash_recheck') }
    foreach ($property in @($readiness.checks.PSObject.Properties)) {
        if ([bool]$property.Value -ne $true) { $issues.Add('check:' + $property.Name) }
    }
    if ($issues.Count -ne 0) { throw ('V25b readiness is stale or failed: ' + ($issues -join ',') + '; path=' + $readinessPath) }
}

function Assert-GpuMemoryHeadroom {
    $nvidiaSmi = Get-Command 'nvidia-smi.exe' -ErrorAction SilentlyContinue
    if ($null -eq $nvidiaSmi) { throw 'nvidia-smi.exe is unavailable; V25b exact replay requires a CUDA headroom audit.' }
    $output = @(& $nvidiaSmi.Source @('--id=0', '--query-gpu=memory.free,memory.total', '--format=csv,noheader,nounits'))
    if ([int]$LASTEXITCODE -ne 0 -or $output.Count -ne 1) { throw 'Unable to query GPU 0 memory with nvidia-smi.' }
    $freeMiB = [int]::Parse((([string]$output[0] -split ',')[0]).Trim(), [System.Globalization.CultureInfo]::InvariantCulture)
    Write-Host ('[V25b] GPU0 free=' + $freeMiB + ' MiB; required=' + $minimumFreeGpuMemoryMiB + ' MiB') -ForegroundColor Cyan
    if ($freeMiB -lt $minimumFreeGpuMemoryMiB) { throw ('V25b CUDA preflight blocked: free GPU memory is below ' + $minimumFreeGpuMemoryMiB + ' MiB.') }
}

function Assert-CompletedAnalysis {
    Assert-File $reportPath 'V25b exact replay report'
    $report = Read-Utf8Json $reportPath
    if ([string]$report.status -ne 'completed_development_exact_replay_gradient_diagnosis_only' -or [string]$report.schema_version -ne 'RADAR-LN3-V25b-known-gate-exact-replay-parent-gradient-decomposition-report-v1') {
        throw 'V25b report has an invalid identity.'
    }
    if ([int]$report.integrity.replayed_step_count -ne 9860 -or [int]$report.integrity.expected_step_count -ne 9860 -or $report.integrity.every_parent_batch_hash_matched_exactly -ne $true -or $report.integrity.every_clip_decision_matched_exactly -ne $true) {
        throw 'V25b report does not prove complete exact replay.'
    }
    if ([int]$report.replay_scope.validation_payload_access_count -ne 0 -or [int]$report.replay_scope.test_payload_access_count -ne 0 -or [int]$report.replay_scope.unknown_payload_access_count -ne 0 -or [bool]$report.replay_scope.checkpoint_written -or [bool]$report.replay_scope.optimizer_state_written -or [bool]$report.replay_scope.checkpoint_selection_performed) {
        throw 'V25b report violates diagnostic-only sealed scope.'
    }
    if ([bool]$report.deployment_eligible) { throw 'V25b report must not be deployment eligible.' }
    if (@($report.per_clutter_batch).Count -le 0) { throw 'V25b report contains no Clutter-batch decomposition.' }
}

function Invoke-Prepare {
    Invoke-RadarPython @('-m', 'py_compile', 'radar_ln3\known_gate_v25b_gradient_decomposition.py', 'build\known_gate_v25b.py', 'report\known_gate_v25b.py', 'smoke\known_gate_v25b_contract_test.py')
    Invoke-RadarPython @('-m', 'build.known_gate_v25b', '--project-root', $projectPath)
    Assert-Readiness
}

function Invoke-Smoke {
    Assert-SourceArtifact
    Assert-Readiness
    Assert-GpuMemoryHeadroom
    Invoke-RadarPython @('-m', 'smoke.known_gate_v25b_contract_test', '--project-root', $projectPath)
    Invoke-RadarPython @('-m', 'smoke.static_isolation_check', '--root', $srcRoot)
}

function Invoke-Analyze {
    Assert-SourceArtifact
    Assert-Readiness
    Assert-GpuMemoryHeadroom
    Invoke-RadarPython @('-m', 'report.known_gate_v25b', '--project-root', $projectPath)
    Assert-CompletedAnalysis
}

function Invoke-Evaluate {
    Assert-Readiness
    Assert-CompletedAnalysis
}

try {
    Assert-File $python 'Project Python interpreter'
    Write-Host ('[V25b] project=' + $projectPath) -ForegroundColor Green
    Write-Host ('[V25b] stage=' + $Stage + '; V25a exact replay per-parent gradient decomposition') -ForegroundColor Green
    Write-Host '[V25b] Read-only diagnosis: no training, checkpoint mutation, validation/test/Unknown/L3/cascade/stable/P95 evaluation, or deployment claim.' -ForegroundColor Green
    switch ($Stage) {
        'prepare' { Invoke-Prepare }
        'smoke' { Invoke-Smoke }
        'analyze' { Invoke-Analyze }
        'evaluate' { Invoke-Evaluate }
        'all' { Invoke-Prepare; Invoke-Smoke; Invoke-Analyze; Invoke-Evaluate }
    }
    Write-Host '[V25b] requested stage completed.' -ForegroundColor Green
}
catch { [Console]::Error.WriteLine('[V25b] pipeline stopped: ' + $_.Exception.Message); exit 1 }
