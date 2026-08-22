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
$artifactPath = Join-Path $projectPath 'artifacts\RADAR-LN3-V23g-cq08-liquid-known-gate-causal-course-consistency-density-weighted-seed43-formal'
$readinessPath = Join-Path $projectPath 'src-v3\processed\report\RADAR-LN3-V23g-known-gate-density-weighted-training-readiness.json'
$reportPath = Join-Path $projectPath 'src-v3\processed\report\RADAR-LN3-V23g-known-gate-density-weighted-paired-validation.json'
$minimumFreeGpuMemoryMiB = 4096

function Invoke-RadarPython {
    param([Parameter(Mandatory = $true)][string[]]$Arguments)
    $runtimeArguments = @('-X', 'utf8') + $Arguments
    $display = $python + ' ' + ($runtimeArguments -join ' ')
    Write-Host ('> ' + $display) -ForegroundColor Cyan
    Push-Location $srcRoot
    try { & $python @runtimeArguments | Out-Host; $code = [int]$LASTEXITCODE }
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
    Assert-File $readinessPath 'V23g readiness'
    $readiness = Read-Utf8Json $readinessPath
    $issues = [System.Collections.Generic.List[string]]::new()
    if ([string]$readiness.status -ne 'passed') { $issues.Add('status') }
    if ([string]$readiness.schema_version -ne 'RADAR-LN3-V23g-known-gate-density-weighted-readiness-v1') { $issues.Add('schema_version') }
    if ([string]$readiness.variant -ne 'v23g_course_prefix_density_weighted') { $issues.Add('variant') }
    if ([int]$readiness.train_parent_count -ne 986 -or [int]$readiness.validation_parent_count -ne 232 -or [int]$readiness.validation_clutter_parent_support -ne 9) { $issues.Add('support') }
    if ([int]$readiness.selected_seed -ne 43) { $issues.Add('seed') }
    if ([bool]$readiness.validation_used_for_readiness_gate) { $issues.Add('validation_used_for_readiness_gate') }
    if ([int]$readiness.test_payload_access_count -ne 0 -or [bool]$readiness.deployment_eligible) { $issues.Add('sealed_scope') }
    foreach ($property in @($readiness.checks.PSObject.Properties)) { if ([bool]$property.Value -ne $true) { $issues.Add('check:' + $property.Name) } }
    if ($issues.Count -ne 0) { throw ('V23g readiness is stale or failed: ' + ($issues -join ',') + '; path=' + $readinessPath) }
}

function Assert-GpuMemoryHeadroom {
    $nvidiaSmi = Get-Command 'nvidia-smi.exe' -ErrorAction SilentlyContinue
    if ($null -eq $nvidiaSmi) { throw 'nvidia-smi.exe is unavailable; V23g training requires a CUDA headroom audit.' }
    $output = @(& $nvidiaSmi.Source @('--id=0', '--query-gpu=memory.free,memory.total', '--format=csv,noheader,nounits'))
    if ([int]$LASTEXITCODE -ne 0 -or $output.Count -ne 1) { throw 'Unable to query GPU 0 memory with nvidia-smi.' }
    $freeMiB = [int]::Parse((([string]$output[0] -split ',')[0]).Trim(), [System.Globalization.CultureInfo]::InvariantCulture)
    Write-Host ('[V23g] GPU0 free=' + $freeMiB + ' MiB; required=' + $minimumFreeGpuMemoryMiB + ' MiB') -ForegroundColor Cyan
    if ($freeMiB -lt $minimumFreeGpuMemoryMiB) { throw ('V23g CUDA preflight blocked: free GPU memory is below ' + $minimumFreeGpuMemoryMiB + ' MiB.') }
}

function Assert-CompletedTreatment {
    $summaryPath = Join-Path $artifactPath 'training-summary.json'
    Assert-File $summaryPath 'V23g training summary'
    Assert-File (Join-Path $artifactPath 'best_model.pt') 'V23g best checkpoint'
    Assert-File (Join-Path $artifactPath 'last_model.pt') 'V23g last checkpoint'
    Assert-File (Join-Path $artifactPath 'sample_detail_information.json') 'V23g sample detail'
    for ($epoch = 1; $epoch -le 20; $epoch++) { Assert-File (Join-Path $artifactPath ('epochs\epoch-' + $epoch.ToString('000') + '.pt')) ('V23g epoch ' + $epoch) }
    $summary = Read-Utf8Json $summaryPath
    if ([string]$summary.status -ne 'completed' -or [string]$summary.variant -ne 'v23g_course_prefix_density_weighted' -or [string]$summary.architecture -ne 'self_developed_cfc_liquid_causal_encoder_density_weighted') { throw ('V23g completed artifact identity is invalid: ' + $summaryPath) }
    if (@($summary.history).Count -ne 20 -or $summary.dataloader_contract.all_epoch_parent_orders_exactly_match_v21e_control -ne $true) { throw 'V23g full 20-epoch parent-order pairing is invalid.' }
    if ($summary.initialization_audit.pre_first_zero_delta_replays_fixed_gain_control -ne $true -or $summary.initialization_audit.first_zero_delta_differs_from_fixed_gain_control -ne $true) { throw 'V23g initialization/density exposure audit is invalid.' }
    if ($summary.test_evaluation -ne 'not_executed' -or $summary.deployment_eligible -ne $false) { throw 'V23g completed scope is not sealed.' }
}

function Invoke-Prepare {
    Invoke-RadarPython @('-m', 'py_compile', 'radar_ln3\model.py', 'radar_ln3\known_gate.py', 'radar_ln3\known_gate_v23g.py', 'build\known_gate_v23g.py', 'train\known_gate_v23g.py', 'report\known_gate_v23g.py', 'smoke\known_gate_v23g_density_contract_test.py')
    Invoke-RadarPython @('-m', 'build.known_gate_v23g', '--project-root', $projectPath)
    Assert-Readiness
}

function Invoke-Smoke {
    Assert-Readiness
    Invoke-RadarPython @('-m', 'smoke.known_gate_v23g_density_contract_test', '--project-root', $projectPath)
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
    if (Test-Path -LiteralPath $archivePath) { throw ('V23g failed-artifact archive path already exists: ' + $archivePath) }
    Move-Item -LiteralPath $artifactPath -Destination $archivePath
    Write-Host ('[V23g] archived non-resumable failed attempt to ' + $archivePath) -ForegroundColor Yellow
    return $true
}

function Invoke-Train {
    Assert-Readiness
    $summaryPath = Join-Path $artifactPath 'training-summary.json'
    $lastCheckpoint = Join-Path $artifactPath 'last_model.pt'
    if (Test-Path -LiteralPath $summaryPath -PathType Leaf) {
        $summary = Read-Utf8Json $summaryPath
        if ([string]$summary.status -eq 'completed') { Assert-CompletedTreatment; Write-Host '[V23g] completed treatment exists; preserving it.' -ForegroundColor Yellow; return }
    }
    $arguments = @('-m', 'train.known_gate_v23g', '--project-root', $projectPath)
    if (Test-Path -LiteralPath $artifactPath -PathType Container) {
        $items = @(Get-ChildItem -LiteralPath $artifactPath -Force)
        if ($items.Count -gt 0) {
            if (Test-Path -LiteralPath $lastCheckpoint -PathType Leaf) { $arguments += @('--resume-checkpoint', $lastCheckpoint); Write-Host ('[V23g] resuming from ' + $lastCheckpoint) -ForegroundColor Yellow }
            elseif (-not (Archive-NonResumableFailedArtifact)) { throw ('V23g artifact has no resumable last_model.pt; preserve it for inspection before retrying: ' + $artifactPath) }
        }
    }
    Assert-GpuMemoryHeadroom
    Invoke-RadarPython $arguments
    Assert-CompletedTreatment
}

function Invoke-Evaluate {
    Assert-CompletedTreatment
    Invoke-RadarPython @('-m', 'report.known_gate_v23g', '--project-root', $projectPath)
    Assert-File $reportPath 'V23g paired development report'
    $report = Read-Utf8Json $reportPath
    if ([string]$report.status -ne 'completed_development_single_factor_only' -or [string]$report.schema_version -ne 'RADAR-LN3-V23g-known-gate-density-weighted-paired-validation-v1' -or $report.gate.advance_to_oof -ne $false -or $report.deployment_eligible -ne $false) { throw ('V23g report violates sealed scope: ' + $reportPath) }
}

try {
    Assert-File $python 'Project Python interpreter'
    Write-Host ('[V23g] project=' + $projectPath) -ForegroundColor Green
    Write-Host ('[V23g] stage=' + $Stage + '; V21e fixed 0.25 control versus self-developed density_weighted treatment; full20epoch') -ForegroundColor Green
    Write-Host '[V23g] OOF, test, Unknown, L3, full cascade, rejection, stable and P95 remain intentionally unexecuted.' -ForegroundColor Green
    switch ($Stage) {
        'prepare' { Invoke-Prepare }
        'smoke' { Invoke-Smoke }
        'train' { Invoke-Train }
        'evaluate' { Invoke-Evaluate }
        'all' { Invoke-Prepare; Invoke-Smoke; Invoke-Train; Invoke-Evaluate }
    }
    Write-Host '[V23g] requested stage completed.' -ForegroundColor Green
}
catch { [Console]::Error.WriteLine('[V23g] pipeline stopped: ' + $_.Exception.Message); exit 1 }
