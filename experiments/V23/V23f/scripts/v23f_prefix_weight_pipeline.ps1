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
$defaultProjectRoot = (Resolve-Path -LiteralPath (Split-Path -Parent $srcRoot)).Path
$projectPath = [System.IO.Path]::GetFullPath(
    $(if ([string]::IsNullOrWhiteSpace($ProjectRoot)) { $defaultProjectRoot } else { $ProjectRoot })
)
$python = Join-Path $projectPath '.venv\Scripts\python.exe'
$artifactPath = Join-Path $projectPath 'artifacts\RADAR-LN3-V23f-cq08-known-gate-official-cfc-positive-dt-fixed-retention-prefix-weight-1-0-seed43-formal'
$readinessPath = Join-Path $projectPath 'src-v3\processed\report\RADAR-LN3-V23f-known-gate-official-cfc-prefix-weight-training-readiness.json'
$reportPath = Join-Path $projectPath 'src-v3\processed\report\RADAR-LN3-V23f-known-gate-official-cfc-prefix-weight-paired-validation.json'
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
    Assert-File $readinessPath 'V23f readiness'
    $readiness = Read-Utf8Json $readinessPath
    $issues = [System.Collections.Generic.List[string]]::new()
    if ([string]$readiness.status -ne 'passed') { $issues.Add('status') }
    if ([string]$readiness.schema_version -ne 'RADAR-LN3-V23f-known-gate-official-cfc-prefix-weight-readiness-v1') { $issues.Add('schema_version') }
    if ([string]$readiness.variant -ne 'official_cfc_positive_dt_fixed_retention_prefix_weight_1_0') { $issues.Add('variant') }
    if ([int]$readiness.train_parent_count -ne 986 -or [int]$readiness.validation_parent_count -ne 232 -or [int]$readiness.validation_clutter_parent_support -ne 9) { $issues.Add('support') }
    if ([int]$readiness.selected_seed -ne 43) { $issues.Add('seed') }
    if ([bool]$readiness.validation_used_for_readiness_gate) { $issues.Add('validation_used_for_readiness_gate') }
    if ([int]$readiness.test_payload_access_count -ne 0) { $issues.Add('test_payload_access_count') }
    if ([bool]$readiness.deployment_eligible) { $issues.Add('deployment_eligible') }
    foreach ($property in @($readiness.checks.PSObject.Properties)) {
        if ([bool]$property.Value -ne $true) { $issues.Add('check:' + $property.Name) }
    }
    if ($issues.Count -ne 0) { throw ('V23f readiness is stale or failed: ' + ($issues -join ',') + '; path=' + $readinessPath) }
}

function Assert-GpuMemoryHeadroom {
    $nvidiaSmi = Get-Command 'nvidia-smi.exe' -ErrorAction SilentlyContinue
    if ($null -eq $nvidiaSmi) { throw 'nvidia-smi.exe is unavailable; V23f formal training requires a CUDA headroom audit.' }
    $output = @(& $nvidiaSmi.Source @('--id=0', '--query-gpu=memory.free,memory.total', '--format=csv,noheader,nounits'))
    if ([int]$LASTEXITCODE -ne 0 -or $output.Count -ne 1) { throw 'Unable to query GPU 0 memory with nvidia-smi.' }
    $parts = @([string]$output[0] -split ',')
    if ($parts.Count -ne 2) { throw ('Unexpected nvidia-smi response: ' + [string]$output[0]) }
    $freeMiB = [int]::Parse($parts[0].Trim(), [System.Globalization.CultureInfo]::InvariantCulture)
    Write-Host ('[V23f] GPU0 free=' + $freeMiB + ' MiB; required=' + $minimumFreeGpuMemoryMiB + ' MiB') -ForegroundColor Cyan
    if ($freeMiB -lt $minimumFreeGpuMemoryMiB) { throw ('V23f CUDA preflight blocked: free GPU memory is below ' + $minimumFreeGpuMemoryMiB + ' MiB.') }
}

function Assert-CompletedTreatment {
    $summaryPath = Join-Path $artifactPath 'training-summary.json'
    $lastCheckpoint = Join-Path $artifactPath 'last_model.pt'
    Assert-File $summaryPath 'V23f training summary'
    Assert-File $lastCheckpoint 'V23f last checkpoint'
    for ($epoch = 1; $epoch -le 20; $epoch++) { Assert-File (Join-Path $artifactPath ('epochs\epoch-' + $epoch.ToString('000') + '.pt')) ('V23f epoch checkpoint ' + $epoch) }
    $summary = Read-Utf8Json $summaryPath
    if ([string]$summary.status -notin @('completed', 'completed_no_feasible_checkpoint')) { throw ('V23f training did not complete: ' + [string]$summary.status) }
    if (@($summary.history).Count -ne 20) { throw 'V23f history is not 20 epochs.' }
    if ($summary.dataloader_contract.all_epoch_parent_orders_exactly_match_v23_control -ne $true) { throw 'V23f parent order pairing failed.' }
    if ($summary.test_evaluation -ne 'not_executed' -or $summary.deployment_eligible -ne $false) { throw 'V23f completed scope is not sealed.' }
    if ([string]$summary.status -eq 'completed') {
        Assert-File (Join-Path $artifactPath 'best_model.pt') 'V23f feasible best checkpoint'
        Assert-File (Join-Path $artifactPath 'sample_detail_information.json') 'V23f feasible sample detail'
    }
}

function Invoke-Prepare {
    Invoke-RadarPython @('-m', 'py_compile', 'radar_ln3\known_gate.py', 'radar_ln3\known_gate_v23e_selection.py', 'radar_ln3\known_gate_v23f.py', 'build\known_gate_v23f.py', 'train\known_gate_v23f.py', 'report\known_gate_v23f.py')
    Invoke-RadarPython @('-m', 'build.known_gate_v23f', '--project-root', $projectPath)
    Assert-Readiness
}


function Invoke-Train {
    Assert-Readiness
    $summaryPath = Join-Path $artifactPath 'training-summary.json'
    $lastCheckpoint = Join-Path $artifactPath 'last_model.pt'
    if (Test-Path -LiteralPath $summaryPath -PathType Leaf) {
        $summary = Read-Utf8Json $summaryPath
        if ([string]$summary.status -in @('completed', 'completed_no_feasible_checkpoint')) {
            Assert-CompletedTreatment
            Write-Host '[V23f] completed treatment exists; preserving it.' -ForegroundColor Yellow
            return
        }
    }
    $arguments = @('-m', 'train.known_gate_v23f', '--project-root', $projectPath)
    if (Test-Path -LiteralPath $artifactPath -PathType Container) {
        $items = @(Get-ChildItem -LiteralPath $artifactPath -Force)
        if ($items.Count -gt 0) {
            if (Test-Path -LiteralPath $lastCheckpoint -PathType Leaf) {
                $arguments += @('--resume-checkpoint', $lastCheckpoint)
                Write-Host ('[V23f] resuming from ' + $lastCheckpoint) -ForegroundColor Yellow
            }
            else { throw ('V23f artifact has no resumable last_model.pt; preserve it for inspection before retrying: ' + $artifactPath) }
        }
    }
    Assert-GpuMemoryHeadroom
    Invoke-RadarPython $arguments
    Assert-CompletedTreatment
}

function Invoke-Evaluate {
    Assert-CompletedTreatment
    Invoke-RadarPython @('-m', 'report.known_gate_v23f', '--project-root', $projectPath)
    Assert-File $reportPath 'V23f development report'
    $report = Read-Utf8Json $reportPath
    if ([string]$report.status -ne 'completed_development_prefix_weight_only' -or $report.deployment_eligible -ne $false) { throw ('V23f report violates scope: ' + $reportPath) }
}

try {
    Assert-File $python 'Project Python interpreter'
    Write-Host ('[V23f] project=' + $projectPath) -ForegroundColor Green
    Write-Host ('[V23f] stage=' + $Stage + '; only prefix_auxiliary_weight=1.0 differs from V23a; full20epoch') -ForegroundColor Green
    Write-Host '[V23f] OOF, test, Unknown, L3, full cascade, rejection, stable and P95 remain intentionally unexecuted.' -ForegroundColor Green
    switch ($Stage) {
        'prepare' { Invoke-Prepare }
        'train' { Invoke-Train }
        'evaluate' { Invoke-Evaluate }
        'all' { Invoke-Prepare; Invoke-Train; Invoke-Evaluate }
    }
    Write-Host '[V23f] requested stage completed.' -ForegroundColor Green
}
catch {
    [Console]::Error.WriteLine('[V23f] pipeline stopped: ' + $_.Exception.Message)
    exit 1
}
