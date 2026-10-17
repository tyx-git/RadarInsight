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
$srcRoot = Split-Path -Parent $scriptRoot
$defaultProjectRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $srcRoot))
$projectPath = [System.IO.Path]::GetFullPath(
    $(if ([string]::IsNullOrWhiteSpace($ProjectRoot)) { $defaultProjectRoot } else { $ProjectRoot })
)
$python = Join-Path $projectPath '.venv\Scripts\python.exe'
$artifactPath = Join-Path $projectPath 'artifacts\RADAR-LN3-V23-cq08-known-gate-official-cfc-baseline-seed43-formal'
$configPath = Join-Path $projectPath 'config\RADAR-LN3\RADAR-LN3-V23-cq08-known-gate-official-cfc-baseline-seed43-config.json'
$readinessPath = Join-Path $projectPath 'src-v3\processed\report\RADAR-LN3-V23-known-gate-official-cfc-baseline-training-readiness.json'
$runtimeCompatibilityPath = Join-Path $projectPath 'src-v3\processed\report\RADAR-LN3-V23-official-cfc-runtime-compatibility.json'
$reportPath = Join-Path $projectPath 'src-v3\processed\report\RADAR-LN3-V23-known-gate-official-cfc-baseline-validation.json'
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
    Assert-File $readinessPath 'V23 baseline readiness'
    Assert-File $configPath 'V23 baseline config'
    $readiness = Read-Utf8Json $readinessPath
    $issues = [System.Collections.Generic.List[string]]::new()
    if ([string]$readiness.status -ne 'passed') { $issues.Add('status') }
    if ([string]$readiness.schema_version -ne 'RADAR-LN3-V23-known-gate-official-cfc-baseline-readiness-v1') { $issues.Add('schema_version') }
    if ([int]$readiness.train_parent_count -ne 986) { $issues.Add('train_parent_count') }
    if ([int]$readiness.validation_parent_count -ne 232) { $issues.Add('validation_parent_count') }
    if ([int]$readiness.validation_clutter_parent_support -ne 9) { $issues.Add('validation_clutter_parent_support') }
    if ([int]$readiness.selected_seed -ne 43) { $issues.Add('selected_seed') }
    if ([bool]$readiness.validation_used_for_readiness_gate) { $issues.Add('validation_used_for_readiness_gate') }
    if ([int]$readiness.test_payload_access_count -ne 0) { $issues.Add('test_payload_access_count') }
    if ([bool]$readiness.checks.five_fold_oof_is_deferred -ne $true) { $issues.Add('check:five_fold_oof_is_deferred') }
    if ([bool]$readiness.deployment_eligible) { $issues.Add('deployment_eligible') }
    foreach ($property in @($readiness.checks.PSObject.Properties)) {
        if ([bool]$property.Value -ne $true) { $issues.Add('check:' + $property.Name) }
    }
    if ($issues.Count -ne 0) {
        throw ('V23 baseline readiness is stale or failed; fields=' + ($issues -join ',') + '; path=' + $readinessPath)
    }
}

function Assert-RuntimeCompatibility {
    Assert-File $runtimeCompatibilityPath 'V23 official-CfC runtime compatibility report'
    $compatibility = Read-Utf8Json $runtimeCompatibilityPath
    $issues = [System.Collections.Generic.List[string]]::new()
    if ([string]$compatibility.status -ne 'passed') { $issues.Add('status') }
    if ([string]$compatibility.schema_version -ne 'RADAR-LN3-V23-official-cfc-runtime-compatibility-v1') { $issues.Add('schema_version') }
    foreach ($field in @('dataset_access_count', 'checkpoint_access_count', 'validation_access_count', 'test_access_count')) {
        if ([int]$compatibility.$field -ne 0) { $issues.Add($field) }
    }
    $requiredChecks = @(
        'exact_ncps_distribution_version',
        'exact_official_runtime_binding',
        'official_cfc_forward_has_no_padding_mask',
        'official_cfc_batch_two_heterogeneous_timespans_are_not_supported',
        'official_cfc_zero_dt_is_finite_but_not_state_identity',
        'official_ltc_zero_dt_is_not_safe_for_v23_direct_swap',
        'official_cfc_baseline_is_dense_not_wired'
    )
    foreach ($name in $requiredChecks) {
        if (-not ($compatibility.checks.PSObject.Properties.Name -contains $name) -or [bool]$compatibility.checks.$name -ne $true) {
            $issues.Add('check:' + $name)
        }
    }
    if (@($compatibility.checks.PSObject.Properties).Count -ne $requiredChecks.Count) { $issues.Add('check_set') }
    if ($issues.Count -ne 0) {
        throw ('V23 official-CfC runtime compatibility is stale or failed; fields=' + ($issues -join ',') + '; path=' + $runtimeCompatibilityPath)
    }
}

function Get-GpuMemoryState {
    $nvidiaSmi = Get-Command 'nvidia-smi.exe' -ErrorAction SilentlyContinue
    if ($null -eq $nvidiaSmi) {
        throw 'nvidia-smi.exe is unavailable; V23 cannot audit CUDA memory headroom.'
    }
    $output = @(& $nvidiaSmi.Source @(
        '--id=0',
        '--query-gpu=memory.free,memory.total',
        '--format=csv,noheader,nounits'
    ))
    if ([int]$LASTEXITCODE -ne 0 -or $output.Count -ne 1) {
        throw 'Unable to query GPU 0 memory with nvidia-smi.'
    }
    $parts = @([string]$output[0] -split ',')
    if ($parts.Count -ne 2) {
        throw ('Unexpected nvidia-smi memory response: ' + [string]$output[0])
    }
    return [pscustomobject]@{
        FreeMiB = [int]::Parse($parts[0].Trim(), [System.Globalization.CultureInfo]::InvariantCulture)
        TotalMiB = [int]::Parse($parts[1].Trim(), [System.Globalization.CultureInfo]::InvariantCulture)
    }
}

function Assert-GpuMemoryHeadroom {
    $memory = Get-GpuMemoryState
    Write-Host ('[V23] GPU0 free=' + $memory.FreeMiB + ' MiB; required=' + $minimumFreeGpuMemoryMiB + ' MiB') -ForegroundColor Cyan
    if ($memory.FreeMiB -lt $minimumFreeGpuMemoryMiB) {
        throw ('V23 CUDA preflight blocked: free GPU memory is below ' + $minimumFreeGpuMemoryMiB + ' MiB.')
    }
}

function Assert-CompletedBaseline {
    $summaryPath = Join-Path $artifactPath 'training-summary.json'
    $checkpointPath = Join-Path $artifactPath 'best_model.pt'
    $lastCheckpoint = Join-Path $artifactPath 'last_model.pt'
    $detailPath = Join-Path $artifactPath 'sample_detail_information.json'
    Assert-File $summaryPath 'V23 training summary'
    Assert-File $checkpointPath 'V23 best checkpoint'
    Assert-File $lastCheckpoint 'V23 last checkpoint'
    Assert-File $detailPath 'V23 sample detail'
    for ($epoch = 1; $epoch -le 20; $epoch++) {
        Assert-File (Join-Path $artifactPath ('epochs\epoch-' + $epoch.ToString('000') + '.pt')) ('V23 epoch checkpoint ' + $epoch)
    }
    $summary = Read-Utf8Json $summaryPath
    $compatibility = Read-Utf8Json $runtimeCompatibilityPath
    $compatibilityHash = (Get-FileHash -LiteralPath $runtimeCompatibilityPath -Algorithm SHA256).Hash.ToLowerInvariant()
    $summaryCompatibilityChecks = @($summary.official_runtime_compatibility.checks.PSObject.Properties)
    $runtimeCompatibilityChecks = @($compatibility.checks.PSObject.Properties)
    $compatibilityChecksMatch = $summaryCompatibilityChecks.Count -eq $runtimeCompatibilityChecks.Count
    if ($compatibilityChecksMatch) {
        $summaryCheckNames = @($summaryCompatibilityChecks | ForEach-Object { [string]$_.Name } | Sort-Object)
        $runtimeCheckNames = @($runtimeCompatibilityChecks | ForEach-Object { [string]$_.Name } | Sort-Object)
        if ([string]::Join("`n", [string[]]$summaryCheckNames) -ne [string]::Join("`n", [string[]]$runtimeCheckNames)) {
            $compatibilityChecksMatch = $false
        }
    }
    if ($compatibilityChecksMatch) {
        foreach ($runtimeCheck in $runtimeCompatibilityChecks) {
            $summaryCheck = $summary.official_runtime_compatibility.checks.PSObject.Properties[$runtimeCheck.Name]
            if ($null -eq $summaryCheck -or [bool]$summaryCheck.Value -ne [bool]$runtimeCheck.Value) {
                $compatibilityChecksMatch = $false
                break
            }
        }
    }
    if (
        $summary.status -ne 'completed' -or
        $summary.task_name -ne 'known_gate' -or
        $summary.variant -ne 'official_cfc_baseline' -or
        $summary.architecture -ne 'official_ncps_cfc' -or
        [int]$summary.seed -ne 43 -or
        [int]$summary.dataloader_seed -ne 43 -or
        [int]$summary.train_parent_count -ne 986 -or
        [int]$summary.validation_parent_count -ne 232 -or
        [int]$summary.validation_clutter_support -ne 9 -or
        $summary.full_parameter_training -ne $true -or
        $summary.all_model_parameters_trainable -ne $true -or
        $summary.model_state_changed_from_initialization -ne $true -or
        -not ($summary.PSObject.Properties.Name -contains 'official_runtime_compatibility') -or
        $summary.official_runtime_compatibility.schema_version -ne 'RADAR-LN3-V23-official-cfc-runtime-compatibility-v1' -or
        $summary.official_runtime_compatibility.path -ne $runtimeCompatibilityPath -or
        $summary.official_runtime_compatibility.sha256 -ne $compatibilityHash -or
        -not $compatibilityChecksMatch -or
        -not ($summary.PSObject.Properties.Name -contains 'OOF') -or
        $summary.test_evaluation -ne 'not_executed' -or
        $summary.deployment_eligible -ne $false
    ) {
        throw ('V23 baseline is incomplete or outside its formal contract: ' + $summaryPath)
    }
}

function Invoke-Prepare {
    Invoke-RadarPython @(
        '-m', 'py_compile',
        'radar_ln3\official_cfc.py',
        'radar_ln3\known_gate_official_cfc.py',
        'radar_ln3\known_gate_v23.py',
        'build\known_gate_v23.py',
        'train\known_gate_v23.py',
        'report\known_gate_v23.py'
    )
    Invoke-RadarPython @('-m', 'build.known_gate_v23', '--project-root', $projectPath)
    Assert-Readiness
    Assert-RuntimeCompatibility
}


function Invoke-Train {
    Assert-Readiness
    Assert-RuntimeCompatibility
    $summaryPath = Join-Path $artifactPath 'training-summary.json'
    $lastCheckpoint = Join-Path $artifactPath 'last_model.pt'
    if (Test-Path -LiteralPath $summaryPath -PathType Leaf) {
        $summary = Read-Utf8Json $summaryPath
        if ($summary.status -eq 'completed') {
            Assert-CompletedBaseline
            Write-Host '[V23] completed baseline exists; preserving it.' -ForegroundColor Yellow
            return
        }
    }
    $arguments = @('-m', 'train.known_gate_v23', '--project-root', $projectPath)
    if (Test-Path -LiteralPath $artifactPath -PathType Container) {
        $items = @(Get-ChildItem -LiteralPath $artifactPath -Force)
        if ($items.Count -gt 0) {
            if (Test-Path -LiteralPath $lastCheckpoint -PathType Leaf) {
                $arguments += @('--resume-checkpoint', $lastCheckpoint)
                Write-Host ('[V23] resuming from ' + $lastCheckpoint) -ForegroundColor Yellow
            }
            else {
                throw ('V23 artifact has no resumable last_model.pt. Preserve it for inspection and manually archive it before retrying: ' + $artifactPath)
            }
        }
    }
    Assert-GpuMemoryHeadroom
    Invoke-RadarPython $arguments
    Assert-CompletedBaseline
}

function Invoke-Evaluate {
    Assert-RuntimeCompatibility
    Assert-CompletedBaseline
    Invoke-RadarPython @('-m', 'report.known_gate_v23', '--project-root', $projectPath)
    Assert-File $reportPath 'V23 development validation report'
    $report = Read-Utf8Json $reportPath
    if (
        $report.status -ne 'completed_development_baseline_only' -or
        $report.schema_version -ne 'RADAR-LN3-V23-known-gate-official-cfc-baseline-validation-report-v1' -or
        $report.next_stage.multiseed_required_before_oof -ne $true -or
        $report.next_stage.deployment_eligible -ne $false
    ) {
        throw ('V23 development validation report violates its contract: ' + $reportPath)
    }
}

try {
    Assert-File $python 'Project Python interpreter'
    Write-Host ('[V23] project=' + $projectPath) -ForegroundColor Green
    Write-Host ('[V23] stage=' + $Stage + '; pure official dense CfC baseline; seed=43') -ForegroundColor Green
    Write-Host '[V23] OOF is intentionally deferred. This command only prepares, trains, and evaluates the development baseline.' -ForegroundColor Green
    switch ($Stage) {
        'prepare' { Invoke-Prepare }
        'train' { Invoke-Train }
        'evaluate' { Invoke-Evaluate }
        'all' {
            Invoke-Prepare
            Invoke-Train
            Invoke-Evaluate
        }
    }
    Write-Host '[V23] requested stage completed. Multiseed, OOF, test, rejection, cascade, stable, P95 and deployment remain unexecuted.' -ForegroundColor Green
}
catch {
    [Console]::Error.WriteLine('[V23] pipeline stopped: ' + $_.Exception.Message)
    exit 1
}
