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
$contractPath = Join-Path $projectPath 'data\processed\RADAR-LN3-V26b-cq08-frozen-l1-l2-raw-routing-contract.json'
$cacheManifestPath = Join-Path $projectPath 'data\processed\RADAR-LN3-V26b-cq08-l2-mixed-open-set-feature-cache-manifest.json'
$registryPath = Join-Path $projectPath 'data\processed\RADAR-LN3-V26b-cq08-semantic-decision-layer-unfitted-registry.json'
$readinessPath = Join-Path $projectPath 'src-v3\processed\report\RADAR-LN3-V26b-frozen-l1-l2-raw-routing-readiness.json'
$reportPath = Join-Path $projectPath 'src-v3\processed\report\RADAR-LN3-V26b-frozen-l1-l2-raw-routing-audit.json'
$detailPath = Join-Path $projectPath 'src-v3\processed\report\RADAR-LN3-V26b-frozen-l1-l2-raw-routing-parent-details.csv'
$minimumFreeGpuMemoryMiB = 4096
$maxPersistedJsonBytes = 2 * 1024 * 1024

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

function Assert-GpuMemoryHeadroom {
    $nvidiaSmi = Get-Command 'nvidia-smi.exe' -ErrorAction SilentlyContinue
    if ($null -eq $nvidiaSmi) { throw 'nvidia-smi.exe is unavailable; V26b inference requires a CUDA headroom audit.' }
    $output = @(& $nvidiaSmi.Source @('--id=0', '--query-gpu=memory.free', '--format=csv,noheader,nounits'))
    if ([int]$LASTEXITCODE -ne 0 -or $output.Count -ne 1) { throw 'Unable to query GPU 0 memory with nvidia-smi.' }
    $freeMiB = [int]::Parse(([string]$output[0]).Trim(), [System.Globalization.CultureInfo]::InvariantCulture)
    Write-Host ('[V26b] GPU0 free=' + $freeMiB + ' MiB; required=' + $minimumFreeGpuMemoryMiB) -ForegroundColor Cyan
    if ($freeMiB -lt $minimumFreeGpuMemoryMiB) { throw ('V26b CUDA preflight blocked: free GPU memory is below ' + $minimumFreeGpuMemoryMiB + ' MiB.') }
}

function Assert-Preparation {
    Assert-File $contractPath 'V26b contract'
    Assert-File $cacheManifestPath 'V26b L2 mixed-cache manifest'
    Assert-File $registryPath 'V26b unfitted L3 registry'
    Assert-File $readinessPath 'V26b readiness'
    $contract = Read-Utf8Json $contractPath
    $cacheManifest = Read-Utf8Json $cacheManifestPath
    $registry = Read-Utf8Json $registryPath
    $readiness = Read-Utf8Json $readinessPath
    if ([string]$contract.schema_version -ne 'RADAR-LN3-V26b-frozen-l1-l2-raw-routing-contract-v1') { throw 'V26b contract schema is invalid.' }
    if ([string]$cacheManifest.schema_version -ne 'RADAR-LN3-V26b-l2-mixed-open-set-feature-cache-v1') { throw 'V26b cache manifest schema is invalid.' }
    if ([string]$registry.schema_version -ne 'RADAR-LN3-semantic-decision-layer-registry-v1' -or [string]$registry.calibration.status -ne 'unfitted_raw_routing_audit_only' -or $registry.final_actions_enabled -ne $false) { throw 'V26b L3 registry is not explicitly unfitted.' }
    if ([string]$readiness.status -ne 'passed' -or $readiness.deployment_eligible -ne $false) { throw 'V26b readiness violates frozen audit scope.' }
}

function Invoke-Prepare {
    Invoke-RadarPython @('-m', 'py_compile', 'build\micro_doppler_v18.py', 'build\known_gate_v26b_l1_l2_raw_routing.py', 'report\known_gate_v26b_l1_l2_raw_routing.py', 'radar_ln3\l1_l2_raw_routing.py', 'radar_ln3\semantic_decision_registry.py', 'smoke\known_gate_v26b_l1_l2_raw_routing_contract_test.py')
    Invoke-RadarPython @('-m', 'build.known_gate_v26b_l1_l2_raw_routing', '--project-root', $projectPath)
    Assert-Preparation
}

function Invoke-Smoke {
    Assert-Preparation
    Invoke-RadarPython @('-m', 'smoke.known_gate_v26b_l1_l2_raw_routing_contract_test')
    Invoke-RadarPython @('-m', 'smoke.static_isolation_check', '--root', $srcRoot)
}

function Invoke-Evaluate {
    Assert-Preparation
    Assert-GpuMemoryHeadroom
    Invoke-RadarPython @('-m', 'report.known_gate_v26b_l1_l2_raw_routing', '--project-root', $projectPath)
    Assert-File $reportPath 'V26b raw-routing report'
    Assert-File $detailPath 'V26b parent detail audit'
    $length = (Get-Item -LiteralPath $reportPath).Length
    if ($length -gt $maxPersistedJsonBytes) { throw ('V26b report exceeds bounded size: ' + $length) }
    $report = Read-Utf8Json $reportPath
    if ([string]$report.status -ne 'completed_frozen_l1_l2_raw_routing_audit' -or $report.deployment_eligible -ne $false -or [string]$report.l3_registry.calibration_status -ne 'unfitted_raw_routing_audit_only') { throw 'V26b report violates raw-routing-only scope.' }
}

try {
    Assert-File $python 'Project Python interpreter'
    Write-Host ('[V26b] project=' + $projectPath) -ForegroundColor Green
    Write-Host ('[V26b] stage=' + $Stage + '; frozen V26 L1 plus V18r L2 raw-routing audit') -ForegroundColor Green
    Write-Host '[V26b] No training, OOF calibration, final L3 threshold, formal Unknown rejection or deployment claim is executed.' -ForegroundColor Green
    switch ($Stage) {
        'prepare' { Invoke-Prepare }
        'smoke' { Invoke-Smoke }
        'evaluate' { Invoke-Evaluate }
        'all' { Invoke-Prepare; Invoke-Smoke; Invoke-Evaluate }
    }
    Write-Host '[V26b] requested stage completed.' -ForegroundColor Green
}
catch { [Console]::Error.WriteLine('[V26b] pipeline stopped: ' + $_.Exception.Message); exit 1 }
