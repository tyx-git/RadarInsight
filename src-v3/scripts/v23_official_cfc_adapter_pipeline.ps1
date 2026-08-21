[CmdletBinding()]
param(
    [ValidateSet('V23a', 'V23b', 'V23c')]
    [string]$Variant,
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
$srcRoot = Split-Path -Parent $scriptRoot
$defaultProjectRoot = Split-Path -Parent $srcRoot
$projectPath = [System.IO.Path]::GetFullPath(
    $(if ([string]::IsNullOrWhiteSpace($ProjectRoot)) { $defaultProjectRoot } else { $ProjectRoot })
)
$python = Join-Path $projectPath '.venv\Scripts\python.exe'
$minimumFreeGpuMemoryMiB = 4096

$variantMap = @{
    'V23a' = [pscustomobject]@{
        Key = 'official_cfc_positive_dt_fixed_retention'
        ArtifactName = 'RADAR-LN3-V23a-cq08-known-gate-official-cfc-positive-dt-fixed-retention-seed43-formal'
        ReadinessName = 'RADAR-LN3-V23a-known-gate-official-cfc-training-readiness.json'
        ReportName = 'RADAR-LN3-V23a-known-gate-official-cfc-paired-validation.json'
        ReadinessSchema = 'RADAR-LN3-V23a-known-gate-official-cfc-adapter-readiness-v1'
        ReportSchema = 'RADAR-LN3-V23a-known-gate-official-cfc-adapter-paired-validation-v1'
    }
    'V23b' = [pscustomobject]@{
        Key = 'official_cfc_recurrent_layer_norm'
        ArtifactName = 'RADAR-LN3-V23b-cq08-known-gate-official-cfc-recurrent-layer-norm-seed43-formal'
        ReadinessName = 'RADAR-LN3-V23b-known-gate-official-cfc-training-readiness.json'
        ReportName = 'RADAR-LN3-V23b-known-gate-official-cfc-paired-validation.json'
        ReadinessSchema = 'RADAR-LN3-V23b-known-gate-official-cfc-adapter-readiness-v1'
        ReportSchema = 'RADAR-LN3-V23b-known-gate-official-cfc-adapter-paired-validation-v1'
    }
    'V23c' = [pscustomobject]@{
        Key = 'official_cfc_zero_delta_bounded_observation'
        ArtifactName = 'RADAR-LN3-V23c-cq08-known-gate-official-cfc-zero-delta-bounded-observation-seed43-formal'
        ReadinessName = 'RADAR-LN3-V23c-known-gate-official-cfc-training-readiness.json'
        ReportName = 'RADAR-LN3-V23c-known-gate-official-cfc-paired-validation.json'
        ReadinessSchema = 'RADAR-LN3-V23c-known-gate-official-cfc-adapter-readiness-v1'
        ReportSchema = 'RADAR-LN3-V23c-known-gate-official-cfc-adapter-paired-validation-v1'
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
        'report\known_gate_v23_adapters.py',
        'smoke\known_gate_v23_adapter_contract_test.py'
    )
    Invoke-RadarPython @('-m', 'build.known_gate_v23_adapters', '--project-root', $projectPath, '--variant', $selected.Key)
    Assert-Readiness
}

function Invoke-Smoke {
    Assert-Readiness
    Invoke-RadarPython @('-m', 'smoke.known_gate_v23_official_cfc_contract_test', '--project-root', $projectPath)
    Invoke-RadarPython @('-m', 'smoke.known_gate_v23_adapter_contract_test', '--project-root', $projectPath, '--variant', $selected.Key)
    Invoke-RadarPython @('-m', 'smoke.static_isolation_check', '--root', $srcRoot)
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
            else {
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
}

try {
    Assert-File $python 'Project Python interpreter'
    Write-Host ('[' + $Variant + '] project=' + $projectPath) -ForegroundColor Green
    Write-Host ('[' + $Variant + '] stage=' + $Stage + '; one isolated external mechanism; seed=43; full20epoch') -ForegroundColor Green
    Write-Host ('[' + $Variant + '] OOF, test, Unknown, L3, full cascade, rejection, stable and P95 remain intentionally unexecuted.') -ForegroundColor Green
    switch ($Stage) {
        'prepare' { Invoke-Prepare }
        'smoke' { Invoke-Smoke }
        'train' { Invoke-Train }
        'evaluate' { Invoke-Evaluate }
        'all' { Invoke-Prepare; Invoke-Smoke; Invoke-Train; Invoke-Evaluate }
    }
    Write-Host ('[' + $Variant + '] requested stage completed.') -ForegroundColor Green
}
catch {
    [Console]::Error.WriteLine('[' + $Variant + '] pipeline stopped: ' + $_.Exception.Message)
    exit 1
}
