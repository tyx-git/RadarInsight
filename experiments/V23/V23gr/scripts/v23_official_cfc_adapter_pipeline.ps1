[CmdletBinding()]
param(
    [ValidateSet('V23gr')]
    [string]$Variant,
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
$srcRoot = Split-Path -Parent $scriptRoot
$defaultProjectRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $srcRoot))
$projectPath = [System.IO.Path]::GetFullPath(
    $(if ([string]::IsNullOrWhiteSpace($ProjectRoot)) { $defaultProjectRoot } else { $ProjectRoot })
)
$python = Join-Path $projectPath '.venv\Scripts\python.exe'
$minimumFreeGpuMemoryMiB = 4096

$variantMap = @{
    'V23gr' = [pscustomobject]@{
        Key = 'official_cfc_zero_delta_density_weighted_multiplicity'
        ArtifactName = 'RADAR-LN3-V23gr-cq08-known-gate-official-cfc-zero-delta-density-weighted-multiplicity-seed43-formal'
        ReadinessName = 'RADAR-LN3-V23gr-known-gate-official-cfc-training-readiness.json'
        ReportName = 'RADAR-LN3-V23gr-known-gate-official-cfc-paired-validation.json'
        ReadinessSchema = 'RADAR-LN3-V23gr-known-gate-official-cfc-adapter-readiness-v1'
        ReportSchema = 'RADAR-LN3-V23gr-known-gate-official-cfc-adapter-paired-validation-v1'
    }
}
$selected = $variantMap[$Variant]
$artifactPath = Join-Path $projectPath ('artifacts\' + $selected.ArtifactName)
$readinessPath = Join-Path $projectPath ('src-v3\processed\report\' + $selected.ReadinessName)
$reportPath = Join-Path $projectPath ('src-v3\processed\report\' + $selected.ReportName)

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
    finally {
        Pop-Location
    }
    if ($exitCode -ne 0) {
        throw ('Python command failed with exit code ' + $exitCode + ': ' + $display)
    }
}

function Assert-File {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Description
    )
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw ($Description + ' is missing: ' + $Path)
    }
}

function Read-Utf8Json {
    param([Parameter(Mandatory = $true)][string]$Path)
    $text = [System.IO.File]::ReadAllText(
        $Path,
        [System.Text.UTF8Encoding]::new($false, $true)
    )
    return ConvertFrom-Json -InputObject $text
}

function Assert-Readiness {
    Assert-File $readinessPath ($Variant + ' readiness')
    $readiness = Read-Utf8Json $readinessPath
    $issues = [System.Collections.Generic.List[string]]::new()
    if ([string]$readiness.status -ne 'passed') { $issues.Add('status') }
    if ([string]$readiness.schema_version -ne [string]$selected.ReadinessSchema) { $issues.Add('schema_version') }
    if ([string]$readiness.variant -ne [string]$selected.Key) { $issues.Add('variant') }
    if ([int]$readiness.train_parent_count -ne 986) { $issues.Add('train_parent_count') }
    if ([int]$readiness.validation_parent_count -ne 232) { $issues.Add('validation_parent_count') }
    if ([int]$readiness.validation_clutter_parent_support -ne 9) { $issues.Add('validation_clutter_parent_support') }
    if ([int]$readiness.selected_seed -ne 43) { $issues.Add('selected_seed') }
    if ([bool]$readiness.validation_used_for_readiness_gate) { $issues.Add('validation_used_for_readiness_gate') }
    if ([int]$readiness.test_payload_access_count -ne 0) { $issues.Add('test_payload_access_count') }
    if ([bool]$readiness.deployment_eligible) { $issues.Add('deployment_eligible') }
    foreach ($property in @($readiness.checks.PSObject.Properties)) {
        if ([bool]$property.Value -ne $true) { $issues.Add('check:' + $property.Name) }
    }
    if ($issues.Count -ne 0) {
        throw ($Variant + ' readiness is stale or failed; fields=' + ($issues -join ',') + '; path=' + $readinessPath)
    }
}

function Get-GpuMemoryState {
    $nvidiaSmi = Get-Command 'nvidia-smi.exe' -ErrorAction SilentlyContinue
    if ($null -eq $nvidiaSmi) {
        throw 'nvidia-smi.exe is unavailable; V23 adapter training requires a CUDA headroom audit.'
    }
    $output = @(& $nvidiaSmi.Source @('--id=0', '--query-gpu=memory.free,memory.total', '--format=csv,noheader,nounits'))
    if ([int]$LASTEXITCODE -ne 0 -or $output.Count -ne 1) {
        throw 'Unable to query GPU 0 memory with nvidia-smi.'
    }
    $parts = @([string]$output[0] -split ',')
    if ($parts.Count -ne 2) { throw ('Unexpected nvidia-smi response: ' + [string]$output[0]) }
    return [pscustomobject]@{
        FreeMiB = [int]::Parse($parts[0].Trim(), [System.Globalization.CultureInfo]::InvariantCulture)
        TotalMiB = [int]::Parse($parts[1].Trim(), [System.Globalization.CultureInfo]::InvariantCulture)
    }
}

function Assert-GpuMemoryHeadroom {
    $memory = Get-GpuMemoryState
    Write-Host ('[' + $Variant + '] GPU0 free=' + $memory.FreeMiB + ' MiB; required=' + $minimumFreeGpuMemoryMiB + ' MiB') -ForegroundColor Cyan
    if ($memory.FreeMiB -lt $minimumFreeGpuMemoryMiB) {
        throw ($Variant + ' CUDA preflight blocked: free GPU memory is below ' + $minimumFreeGpuMemoryMiB + ' MiB.')
    }
}

function Assert-CompletedTreatment {
    $summaryPath = Join-Path $artifactPath 'training-summary.json'
    $checkpointPath = Join-Path $artifactPath 'best_model.pt'
    $lastCheckpoint = Join-Path $artifactPath 'last_model.pt'
    $detailPath = Join-Path $artifactPath 'sample_detail_information.json'
    Assert-File $summaryPath ($Variant + ' training summary')
    Assert-File $checkpointPath ($Variant + ' best checkpoint')
    Assert-File $lastCheckpoint ($Variant + ' last checkpoint')
    Assert-File $detailPath ($Variant + ' sample detail')
    for ($epoch = 1; $epoch -le 20; $epoch++) {
        Assert-File (Join-Path $artifactPath ('epochs\epoch-' + $epoch.ToString('000') + '.pt')) ($Variant + ' epoch checkpoint ' + $epoch)
    }
    $summary = Read-Utf8Json $summaryPath
    $issues = [System.Collections.Generic.List[string]]::new()
    if ([string]$summary.status -ne 'completed') { $issues.Add('status') }
    if ([string]$summary.variant -ne [string]$selected.Key) { $issues.Add('variant') }
    if ([string]$summary.architecture -ne 'official_ncps_cfc_external_state_adapter') { $issues.Add('architecture') }
    if ([int]$summary.seed -ne 43 -or [int]$summary.dataloader_seed -ne 43) { $issues.Add('seed') }
    if ([int]$summary.train_parent_count -ne 986 -or [int]$summary.validation_parent_count -ne 232 -or [int]$summary.validation_clutter_support -ne 9) { $issues.Add('support') }
    if (@($summary.history).Count -ne 20) { $issues.Add('history') }
    if ($summary.full_parameter_training -ne $true -or $summary.all_model_parameters_trainable -ne $true) { $issues.Add('full_parameter_training') }
    if ($summary.dataloader_contract.all_epoch_parent_orders_exactly_match_v23_control -ne $true) { $issues.Add('parent_order_pairing') }
    if ($Variant -eq 'V23d') {
        if ([int]$summary.v23_initialization_pairing.adapter_trainable_parameter_count -ne 48) { $issues.Add('v23d_adapter_parameter_count') }
        if ($summary.v23_initialization_pairing.v23d_direct_v23a_replay.before_optimizer_step_replays_v23a_within_float32_tolerance -ne $true) { $issues.Add('v23d_direct_v23a_replay') }
        if ($null -eq $summary.v23d_per_dimension_tau_audit) { $issues.Add('v23d_tau_audit') }
        elseif ([int]$summary.v23d_per_dimension_tau_audit.dimension_count -ne 48 -or $summary.v23d_per_dimension_tau_audit.all_tau_positive_and_finite -ne $true) { $issues.Add('v23d_tau_audit') }
    }
    if ($Variant -eq 'V23b-r') {
        if ([int]$summary.v23_initialization_pairing.adapter_trainable_parameter_count -ne 48) { $issues.Add('v23br_adapter_parameter_count') }
        if ($summary.v23_initialization_pairing.v23br_direct_v23b_scale_only_audit.before_optimizer_step_replays_v23b_within_float32_tolerance -ne $true) { $issues.Add('v23br_direct_v23b_replay') }
        if ($summary.external_state_adapter.bias -ne $false) { $issues.Add('v23br_bias') }
    }
    if ($Variant -eq 'V23gr') {
        if ([int]$summary.v23_initialization_pairing.adapter_trainable_parameter_count -ne 0) { $issues.Add('v23gr_adapter_parameter_count') }
        if ($summary.v23_initialization_pairing.v23gr_direct_v23c_multiplicity_audit.pre_zero_delta_replays_v23c_within_float32_tolerance -ne $true) { $issues.Add('v23gr_pre_zero_replay') }
        if ($summary.v23_initialization_pairing.v23gr_direct_v23c_multiplicity_audit.first_k2_zero_delta_differs_from_v23c -ne $true) { $issues.Add('v23gr_first_density_difference') }
    }
    if ($Variant -eq 'V23h') {
        if ($null -eq $summary.v23_initialization_pairing.v23h_direct_v23a_cap_audit) { $issues.Add('v23h_cap_audit') }
        elseif ($summary.v23_initialization_pairing.v23h_direct_v23a_cap_audit.pre_above_cap_positive_dt_replays_v23a_within_float32_tolerance -ne $true -or $summary.v23_initialization_pairing.v23h_direct_v23a_cap_audit.first_real_above_cap_positive_dt_diverges_from_v23a -ne $true) { $issues.Add('v23h_cap_audit') }
        if ([string]$summary.time_policy.kind -ne 'positive_dt_capped' -or [double]$summary.time_policy.positive_dt_cap_seconds -ne 120.0) { $issues.Add('v23h_time_policy') }
    }
    foreach ($row in @($summary.history)) {
        if ($row.parent_order_exactly_matches_v23_control -ne $true -or [string]$row.parent_order_sha256 -ne [string]$row.v23_control_parent_order_sha256) { $issues.Add('epoch_parent_order') }
    }
    if ($summary.test_evaluation -ne 'not_executed' -or $summary.deployment_eligible -ne $false) { $issues.Add('sealed_scope') }
    if ($issues.Count -ne 0) {
        throw ($Variant + ' treatment is incomplete or outside its formal contract: ' + ($issues -join ',') + '; path=' + $summaryPath)
    }
}

function Invoke-Prepare {
    Invoke-RadarPython @(
        '-m', 'py_compile',
        'radar_ln3\official_cfc.py',
        'radar_ln3\official_cfc_state_adapters.py',
        'radar_ln3\known_gate_official_cfc.py',
        'radar_ln3\known_gate_v23_adapters.py',
        'build\known_gate_v23_adapters.py',
        'train\known_gate_v23_adapters.py',
        'report\known_gate_v23_adapters.py'
    )
    Invoke-RadarPython @('-m', 'build.known_gate_v23_adapters', '--project-root', $projectPath, '--variant', $selected.Key)
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
    if (Test-Path -LiteralPath $archivePath) {
        throw ($Variant + ' failed-artifact archive path already exists: ' + $archivePath)
    }
    Move-Item -LiteralPath $artifactPath -Destination $archivePath
    Write-Host ('[' + $Variant + '] archived non-resumable failed attempt to ' + $archivePath) -ForegroundColor Yellow
    return $true
}

function Invoke-Train {
    Assert-Readiness
    $summaryPath = Join-Path $artifactPath 'training-summary.json'
    $lastCheckpoint = Join-Path $artifactPath 'last_model.pt'
    if (Test-Path -LiteralPath $summaryPath -PathType Leaf) {
        $summary = Read-Utf8Json $summaryPath
        if ([string]$summary.status -eq 'completed') {
            Assert-CompletedTreatment
            Write-Host ('[' + $Variant + '] completed treatment exists; preserving it.') -ForegroundColor Yellow
            return
        }
    }
    $arguments = @('-m', 'train.known_gate_v23_adapters', '--project-root', $projectPath, '--variant', $selected.Key)
    if (Test-Path -LiteralPath $artifactPath -PathType Container) {
        $items = @(Get-ChildItem -LiteralPath $artifactPath -Force)
        if ($items.Count -gt 0) {
            if (Test-Path -LiteralPath $lastCheckpoint -PathType Leaf) {
                $arguments += @('--resume-checkpoint', $lastCheckpoint)
                Write-Host ('[' + $Variant + '] resuming from ' + $lastCheckpoint) -ForegroundColor Yellow
            }
            elseif (-not (Archive-NonResumableFailedArtifact)) {
                throw ($Variant + ' artifact has no resumable last_model.pt; preserve it for inspection and archive it manually before retrying: ' + $artifactPath)
            }
        }
    }
    Assert-GpuMemoryHeadroom
    Invoke-RadarPython $arguments
    Assert-CompletedTreatment
}

function Invoke-Evaluate {
    Assert-CompletedTreatment
    Invoke-RadarPython @('-m', 'report.known_gate_v23_adapters', '--project-root', $projectPath, '--variant', $selected.Key)
    Assert-File $reportPath ($Variant + ' paired development report')
    $report = Read-Utf8Json $reportPath
    if (
        [string]$report.status -ne 'completed_development_single_factor_only' -or
        [string]$report.schema_version -ne [string]$selected.ReportSchema -or
        $report.gate.advance_to_oof -ne $false -or
        $report.deployment_eligible -ne $false
    ) {
        throw ($Variant + ' paired development report violates its contract: ' + $reportPath)
    }
    if ($Variant -eq 'V23d') {
        if ($null -eq $report.v23d_vs_v23a_direct_control) {
            throw ($Variant + ' report lacks the required V23a direct-control comparison: ' + $reportPath)
        }
        if ($report.gate.v23a_direct_control_primary.advance_to_oof -ne $false) {
            throw ($Variant + ' report incorrectly advances to OOF: ' + $reportPath)
        }
    }
    if ($Variant -eq 'V23b-r') {
        if ($null -eq $report.v23br_vs_v23b_direct_control) {
            throw ($Variant + ' report lacks the required V23b direct-control comparison: ' + $reportPath)
        }
        if ($report.gate.v23b_direct_control_primary.advance_to_oof -ne $false) {
            throw ($Variant + ' report incorrectly advances to OOF: ' + $reportPath)
        }
    }
    if ($Variant -eq 'V23gr') {
        if ($null -eq $report.v23gr_vs_v23c_direct_control) {
            throw ($Variant + ' report lacks the required V23c direct-control comparison: ' + $reportPath)
        }
        if ($report.gate.v23c_direct_control_primary.advance_to_oof -ne $false) {
            throw ($Variant + ' report incorrectly advances to OOF: ' + $reportPath)
        }
    }
    if ($Variant -eq 'V23h') {
        if ($null -eq $report.v23h_vs_v23a_direct_control) {
            throw ($Variant + ' report lacks the required V23a direct-control comparison: ' + $reportPath)
        }
        if ($report.gate.v23a_direct_control_primary.advance_to_three_seed_confirmation -ne $false -or $report.gate.v23a_direct_control_primary.cap_effect_is_estimable_on_development_validation -ne $false) {
            throw ($Variant + ' report incorrectly claims development cap estimability or advances: ' + $reportPath)
        }
    }
}

try {
    Assert-File $python 'Project Python interpreter'
    Write-Host ('[' + $Variant + '] project=' + $projectPath) -ForegroundColor Green
    Write-Host ('[' + $Variant + '] stage=' + $Stage + '; one isolated external mechanism; seed=43; full20epoch') -ForegroundColor Green
    Write-Host ('[' + $Variant + '] OOF, test, Unknown, L3, full cascade, rejection, stable and P95 remain intentionally unexecuted.') -ForegroundColor Green
    switch ($Stage) {
        'prepare' { Invoke-Prepare }
        'train' { Invoke-Train }
        'evaluate' { Invoke-Evaluate }
        'all' { Invoke-Prepare; Invoke-Train; Invoke-Evaluate }
    }
    Write-Host ('[' + $Variant + '] requested stage completed.') -ForegroundColor Green
}
catch {
    [Console]::Error.WriteLine('[' + $Variant + '] pipeline stopped: ' + $_.Exception.Message)
    exit 1
}
