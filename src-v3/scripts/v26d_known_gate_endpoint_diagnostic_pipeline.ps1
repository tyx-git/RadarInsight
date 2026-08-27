[CmdletBinding()]
param(
    [ValidateSet('prepare', 'smoke', 'evaluate', 'all')]
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
$contractPath = Join-Path $projectPath 'data\processed\RADAR-LN3-V26d-cq08-l1-epoch20-endpoint-diagnostic-contract.json'
$readinessPath = Join-Path $projectPath 'src-v3\processed\report\RADAR-LN3-V26d-l1-epoch20-endpoint-diagnostic-readiness.json'
$reportPath = Join-Path $projectPath 'src-v3\processed\report\RADAR-LN3-V26d-l1-epoch20-endpoint-diagnostic.json'
$detailPath = Join-Path $projectPath 'src-v3\processed\report\RADAR-LN3-V26d-l1-epoch20-endpoint-parent-actions.csv'
$minimumFreeGpuMemoryMiB = 4096
$maxReportBytes = 4 * 1024 * 1024
$maxDetailBytes = 2 * 1024 * 1024

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
    finally {
        Pop-Location
    }
    if ($code -ne 0) { throw ('Python command failed with exit code ' + $code + ': ' + $display) }
}

function Assert-File {
    param([Parameter(Mandatory = $true)][string]$Path, [Parameter(Mandatory = $true)][string]$Description)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { throw ($Description + ' is missing: ' + $Path) }
}

function Read-Utf8Json {
    param([Parameter(Mandatory = $true)][string]$Path)
    return (ConvertFrom-Json -InputObject ([System.IO.File]::ReadAllText($Path, [System.Text.UTF8Encoding]::new($false, $true))))
}

function Assert-Preparation {
    Assert-File $contractPath 'V26d contract'
    Assert-File $readinessPath 'V26d readiness'
    $contract = Read-Utf8Json $contractPath
    $readiness = Read-Utf8Json $readinessPath
    $checkValues = @($readiness.checks.PSObject.Properties | ForEach-Object { $_.Value })
    $failedChecks = @($checkValues | Where-Object { $_ -ne $true })
    if (
        [string]$contract.schema_version -ne 'RADAR-LN3-V26d-l1-epoch20-endpoint-diagnostic-contract-v1' -or
        [string]$readiness.schema_version -ne 'RADAR-LN3-V26d-l1-epoch20-endpoint-diagnostic-readiness-v1' -or
        [string]$readiness.status -ne 'passed' -or
        $contract.checkpoint_protocol.outer_held_out_checkpoint_selection -ne $false -or
        @($contract.folds).Count -ne 5 -or
        @($contract.folds | Where-Object { $_.control_epoch -ne 14 -or $_.treatment_epoch -ne 20 }).Count -ne 0 -or
        $contract.scope.training_executed -ne $false -or
        $readiness.training_executed -ne $false -or
        $readiness.deployment_eligible -ne $false -or
        $checkValues.Count -eq 0 -or
        $failedChecks.Count -ne 0
    ) {
        throw 'V26d preparation violates the frozen read-only endpoint protocol.'
    }
}

function Assert-GpuMemoryHeadroom {
    $nvidiaSmi = Get-Command 'nvidia-smi.exe' -ErrorAction SilentlyContinue
    if ($null -eq $nvidiaSmi) { throw 'nvidia-smi.exe is unavailable; V26d endpoint inference requires a CUDA headroom audit.' }
    $output = @(& $nvidiaSmi.Source @('--id=0', '--query-gpu=memory.free', '--format=csv,noheader,nounits'))
    if ([int]$LASTEXITCODE -ne 0 -or $output.Count -ne 1) { throw 'Unable to query GPU 0 memory with nvidia-smi.' }
    $freeMiB = [int]::Parse(([string]$output[0]).Trim(), [System.Globalization.CultureInfo]::InvariantCulture)
    Write-Host ('[V26d] GPU0 free=' + $freeMiB + ' MiB; required=' + $minimumFreeGpuMemoryMiB) -ForegroundColor Cyan
    if ($freeMiB -lt $minimumFreeGpuMemoryMiB) { throw ('V26d CUDA preflight blocked: free GPU memory is below ' + $minimumFreeGpuMemoryMiB + ' MiB.') }
}

function Invoke-Prepare {
    Invoke-RadarPython @('-m', 'py_compile', 'build\known_gate_v26d_endpoint.py', 'report\known_gate_v26d_endpoint.py', 'radar_ln3\known_gate_v26d_endpoint.py', 'smoke\known_gate_v26d_endpoint_contract_test.py')
    Invoke-RadarPython @('-m', 'build.known_gate_v26d_endpoint', '--project-root', $projectPath)
    Assert-Preparation
}

function Invoke-Smoke {
    Assert-Preparation
    Invoke-RadarPython @('-m', 'smoke.known_gate_v26d_endpoint_contract_test', '--project-root', $projectPath)
    Invoke-RadarPython @('-m', 'smoke.static_isolation_check', '--root', $srcRoot)
}

function Invoke-Evaluate {
    Assert-Preparation
    Assert-GpuMemoryHeadroom
    Invoke-RadarPython @('-m', 'report.known_gate_v26d_endpoint', '--project-root', $projectPath)
    Assert-File $reportPath 'V26d endpoint report'
    Assert-File $detailPath 'V26d parent action audit'
    $reportBytes = (Get-Item -LiteralPath $reportPath).Length
    $detailBytes = (Get-Item -LiteralPath $detailPath).Length
    if ($reportBytes -gt $maxReportBytes) { throw ('V26d report exceeds bounded size: ' + $reportBytes) }
    if ($detailBytes -gt $maxDetailBytes) { throw ('V26d parent detail exceeds bounded size: ' + $detailBytes) }
    $report = Read-Utf8Json $reportPath
    if (
        [string]$report.status -ne 'completed_v26d_l1_epoch20_endpoint_diagnostic' -or
        $report.training_executed -ne $false -or
        $report.deployment_eligible -ne $false -or
        $report.integrity.control_reproduces_v26c_metrics -ne $true -or
        $report.integrity.control_reproduces_v26c_thresholds -ne $true -or
        $report.unknown_training_exposure -ne 0 -or
        $report.unknown_threshold_fit_exposure -ne 0 -or
        $report.validation_access_count -ne 0 -or
        $report.test_access_count -ne 0
    ) {
        throw 'V26d report violates the read-only endpoint diagnostic scope.'
    }
}

try {
    Assert-File $python 'Project Python interpreter'
    Write-Host ('[V26d] project=' + $projectPath) -ForegroundColor Green
    Write-Host ('[V26d] stage=' + $Stage + '; V26c epoch-14 control versus same-trajectory epoch-20 L1 endpoint') -ForegroundColor Green
    Write-Host '[V26d] No training, per-fold checkpoint selection, Unknown, validation, test, full cascade or deployment claim is executed.' -ForegroundColor Green
    switch ($Stage) {
        'prepare' { Invoke-Prepare }
        'smoke' { Invoke-Smoke }
        'evaluate' { Invoke-Evaluate }
        'all' { Invoke-Prepare; Invoke-Smoke; Invoke-Evaluate }
    }
    Write-Host '[V26d] requested stage completed.' -ForegroundColor Green
}
catch {
    [Console]::Error.WriteLine('[V26d] pipeline stopped: ' + $_.Exception.Message)
    exit 1
}
