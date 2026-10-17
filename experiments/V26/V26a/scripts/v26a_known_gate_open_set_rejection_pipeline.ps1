[CmdletBinding()]
param(
    [ValidateSet('prepare', 'validation', 'test', 'evaluate', 'all')]
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
$contractPath = Join-Path $projectPath 'data\processed\RADAR-LN3-V26a-cq08-known-gate-frozen-open-set-rejection-contract.json'
$reportPath = Join-Path $projectPath 'src-v3\processed\report\RADAR-LN3-V26a-known-gate-frozen-open-set-rejection-diagnostic.json'
$scanPath = Join-Path $projectPath 'src-v3\processed\report\RADAR-LN3-V26a-known-gate-frozen-open-set-rejection-threshold-scan.csv'
$selectionPath = Join-Path $projectPath 'src-v3\processed\report\RADAR-LN3-V26a-known-gate-frozen-open-set-rejection-selection.json'
$validationDetailPath = Join-Path $projectPath 'src-v3\processed\report\RADAR-LN3-V26a-known-gate-frozen-open-set-rejection-validation-parent-details.csv'
$testDetailPath = Join-Path $projectPath 'src-v3\processed\report\RADAR-LN3-V26a-known-gate-frozen-open-set-rejection-test-parent-details.csv'
$minimumFreeGpuMemoryMiB = 4096
$maxPersistedJsonBytes = 8 * 1024 * 1024

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

function Assert-BoundedArtifact {
    param([Parameter(Mandatory = $true)][string]$Path, [Parameter(Mandatory = $true)][int64]$MaxBytes)
    Assert-File $Path 'V26a bounded artifact'
    $length = (Get-Item -LiteralPath $Path).Length
    if ($length -gt $MaxBytes) { throw ('V26a artifact exceeds its bounded size: ' + $Path + ' size=' + $length + ' max=' + $MaxBytes) }
}

function Assert-Contract {
    Assert-File $contractPath 'V26a contract'
    $contract = Read-Utf8Json $contractPath
    if ([string]$contract.schema_version -ne 'RADAR-LN3-V26a-known-gate-frozen-open-set-rejection-v1') { throw 'V26a contract schema is invalid.' }
    if ([string]$contract.internal_variant -ne 'v26a_frozen_known_gate_threshold_rejection') { throw 'V26a contract variant is invalid.' }
    if ($contract.variable.training_executed -ne $false -or $contract.variable.model_parameters_changed -ne $false) { throw 'V26a unexpectedly changes training/model parameters.' }
    if ($contract.threshold_selection.test_used_for_selection -ne $false) { throw 'V26a test selection isolation is invalid.' }
    if ([int]$contract.variable.threshold_grid.Count -ne 50) { throw 'V26a threshold grid count is invalid.' }
}

function Assert-GpuMemoryHeadroom {
    $nvidiaSmi = Get-Command 'nvidia-smi.exe' -ErrorAction SilentlyContinue
    if ($null -eq $nvidiaSmi) { throw 'nvidia-smi.exe is unavailable; V26a inference requires a CUDA headroom audit.' }
    $output = @(& $nvidiaSmi.Source @('--id=0', '--query-gpu=memory.free,memory.total', '--format=csv,noheader,nounits'))
    if ([int]$LASTEXITCODE -ne 0 -or $output.Count -ne 1) { throw 'Unable to query GPU 0 memory with nvidia-smi.' }
    $freeMiB = [int]::Parse((([string]$output[0] -split ',')[0]).Trim(), [System.Globalization.CultureInfo]::InvariantCulture)
    Write-Host ('[V26a] GPU0 free=' + $freeMiB + ' MiB; required=' + $minimumFreeGpuMemoryMiB) -ForegroundColor Cyan
    if ($freeMiB -lt $minimumFreeGpuMemoryMiB) { throw ('V26a CUDA preflight blocked: free GPU memory is below ' + $minimumFreeGpuMemoryMiB + ' MiB.') }
}

function Invoke-Prepare {
    Invoke-RadarPython @('-m', 'py_compile', 'build\known_gate_v26a.py', 'report\known_gate_v26a.py', 'radar_ln3\known_gate.py', 'report\known_gate_v26.py')
    Invoke-RadarPython @('-m', 'build.known_gate_v26a', '--project-root', $projectPath)
    Assert-Contract
}


function Invoke-Evaluate {
    Assert-Contract
    Assert-GpuMemoryHeadroom
    Invoke-RadarPython @('-m', 'report.known_gate_v26a', '--project-root', $projectPath, '--phase', 'all')
    Assert-File $reportPath 'V26a report'
    Assert-File $scanPath 'V26a threshold scan'
    Assert-File $selectionPath 'V26a immutable validation selection'
    Assert-File $validationDetailPath 'V26a validation parent detail'
    Assert-File $testDetailPath 'V26a test parent detail'
    Assert-BoundedArtifact $reportPath $maxPersistedJsonBytes
    Assert-BoundedArtifact $selectionPath $maxPersistedJsonBytes
    $report = Read-Utf8Json $reportPath
    if ([string]$report.status -ne 'completed_frozen_open_set_rejection_diagnostic' -or $report.deployment_eligible -ne $false) { throw 'V26a report violates frozen diagnostic scope.' }
    if ($report.test.used_for_threshold_selection -ne $false -or $report.access_audit.test_used_for_threshold_selection -ne $false) { throw 'V26a test isolation failed.' }
}

function Invoke-Validation {
    Assert-Contract
    Assert-GpuMemoryHeadroom
    Invoke-RadarPython @('-m', 'report.known_gate_v26a', '--project-root', $projectPath, '--phase', 'validation')
    Assert-File $selectionPath 'V26a immutable validation selection'
    Assert-File $scanPath 'V26a threshold scan'
    Assert-File $validationDetailPath 'V26a validation parent detail'
    Assert-BoundedArtifact $selectionPath $maxPersistedJsonBytes
    $selection = Read-Utf8Json $selectionPath
    if ([string]$selection.status -ne 'completed_validation_threshold_selection' -or $selection.selection_used_test -ne $false) { throw 'V26a validation selection violates test isolation.' }
}

function Invoke-Test {
    Assert-Contract
    Assert-File $selectionPath 'V26a immutable validation selection'
    Assert-GpuMemoryHeadroom
    Invoke-RadarPython @('-m', 'report.known_gate_v26a', '--project-root', $projectPath, '--phase', 'test')
    Assert-File $reportPath 'V26a report'
    Assert-File $testDetailPath 'V26a test parent detail'
    Assert-BoundedArtifact $reportPath $maxPersistedJsonBytes
    $report = Read-Utf8Json $reportPath
    if ([string]$report.status -ne 'completed_frozen_open_set_rejection_diagnostic' -or $report.test.used_for_threshold_selection -ne $false) { throw 'V26a test audit violates frozen selection scope.' }
}

try {
    Assert-File $python 'Project Python interpreter'
    Write-Host ('[V26a] project=' + $projectPath) -ForegroundColor Green
    Write-Host ('[V26a] stage=' + $Stage + '; frozen V26 checkpoint plus post-checkpoint open-set rejection only') -ForegroundColor Green
    Write-Host '[V26a] No training, feature rebuild, model mutation, Unknown class, L2/L3, full cascade or deployment claim is allowed.' -ForegroundColor Green
    switch ($Stage) {
        'prepare' { Invoke-Prepare }
        'validation' { Invoke-Validation }
        'test' { Invoke-Test }
        'evaluate' { Invoke-Evaluate }
        'all' { Invoke-Prepare; Invoke-Validation; Invoke-Test }
    }
    Write-Host '[V26a] requested stage completed.' -ForegroundColor Green
}
catch { [Console]::Error.WriteLine('[V26a] pipeline stopped: ' + $_.Exception.Message); exit 1 }
