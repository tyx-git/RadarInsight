[CmdletBinding()]
param(
    [ValidateSet('prepare', 'smoke', 'analyze', 'all')]
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
$contractPath = Join-Path $projectPath 'data\processed\RADAR-LN3-V25f-cq08-known-gate-causal-prefix-checkpoint-utility-selection-contract.json'
$readinessPath = Join-Path $projectPath 'src-v3\processed\report\RADAR-LN3-V25f-known-gate-causal-prefix-checkpoint-utility-selection-readiness.json'
$reportPath = Join-Path $projectPath 'src-v3\processed\report\RADAR-LN3-V25f-known-gate-causal-prefix-checkpoint-utility-selection.json'
$sourceArtifactPath = Join-Path $projectPath 'artifacts\RADAR-LN3-V25a-cq08-liquid-known-gate-course-preclip-gradient-telemetry-seed43-formal'

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

function Assert-NoV25fTemporaryFiles {
    $exactTemporaryPaths = @(
        ($contractPath + '.tmp'),
        ($readinessPath + '.tmp'),
        ($reportPath + '.tmp')
    )
    $residual = @(
        $exactTemporaryPaths | Where-Object { Test-Path -LiteralPath $_ -PathType Leaf }
    )
    if ($residual.Count -gt 0) {
        throw ('V25f residual temporary artifact(s) found; remove or recover them before continuing: ' + ($residual -join '; '))
    }
    if (Test-Path -LiteralPath $sourceArtifactPath -PathType Container) {
        $sourceResidual = @(
            Get-ChildItem -LiteralPath $sourceArtifactPath -Recurse -Force -File |
                Where-Object { $_.Name -match '\.(tmp|partial|part|lock)$' }
        )
        if ($sourceResidual.Count -gt 0) {
            throw ('V25f frozen V25a artifact contains residual temporary file(s): ' + (($sourceResidual | ForEach-Object { $_.FullName }) -join '; '))
        }
    }
}

function Assert-Readiness {
    Assert-File $contractPath 'V25f contract'
    Assert-File $readinessPath 'V25f readiness'
    Assert-NoV25fTemporaryFiles
    $contract = Read-Utf8Json $contractPath
    $readiness = Read-Utf8Json $readinessPath
    if ([string]$contract.schema_version -ne 'RADAR-LN3-V25f-known-gate-causal-prefix-checkpoint-utility-selection-experiment-contract-v1' -or [string]$contract.selector_schema -ne 'RADAR-LN3-V25f-causal-prefix-checkpoint-utility-selector-v1' -or [string]$readiness.status -ne 'passed' -or [string]$readiness.selector_schema -ne 'RADAR-LN3-V25f-causal-prefix-checkpoint-utility-selector-v1' -or [string]$readiness.training -ne 'not_executed' -or [int]$readiness.test_payload_access_count -ne 0 -or [int]$readiness.unknown_payload_access_count -ne 0) {
        throw ('V25f readiness is stale or outside selector-only scope: ' + $readinessPath)
    }
    foreach ($property in @($readiness.checks.PSObject.Properties)) {
        if ([bool]$property.Value -ne $true) { throw ('V25f readiness check failed: ' + $property.Name) }
    }
}

function Invoke-Prepare {
    Invoke-RadarPython @('-m', 'py_compile', 'radar_ln3\known_gate.py', 'radar_ln3\known_gate_v25f_selection.py', 'build\known_gate_v25a.py', 'build\known_gate_v25f.py', 'report\known_gate_v25f.py', 'smoke\known_gate_v25f_contract_test.py')
    Invoke-RadarPython @('-m', 'build.known_gate_v25f', '--project-root', $projectPath)
    Assert-Readiness
    Assert-NoV25fTemporaryFiles
}

function Invoke-Smoke {
    Assert-Readiness
    Invoke-RadarPython @('-m', 'smoke.known_gate_v25f_contract_test', '--project-root', $projectPath)
    Invoke-RadarPython @('-m', 'smoke.static_isolation_check', '--root', $srcRoot)
}

function Invoke-Analyze {
    Assert-Readiness
    Invoke-RadarPython @('-m', 'report.known_gate_v25f', '--project-root', $projectPath)
    Assert-File $reportPath 'V25f selector report'
    Assert-NoV25fTemporaryFiles
    $report = Read-Utf8Json $reportPath
    if ([string]$report.status -ne 'completed_development_selector_only' -or [string]$report.selector_schema -ne 'RADAR-LN3-V25f-causal-prefix-checkpoint-utility-selector-v1' -or [int]$report.epoch_candidates.Count -ne 20 -or [int]$report.test_payload_access_count -ne 0 -or [int]$report.unknown_payload_access_count -ne 0 -or [string]$report.test -ne 'not_executed' -or [string]$report.OOF -ne 'not_executed' -or [bool]$report.deployment_eligible) {
        throw ('V25f report violates the frozen selector-only scope: ' + $reportPath)
    }
}

try {
    Assert-File $python 'Project Python interpreter'
    Write-Host ('[V25f] project=' + $projectPath) -ForegroundColor Green
    Write-Host ('[V25f] stage=' + $Stage + '; frozen V25a causal-prefix checkpoint utility scan; no retraining') -ForegroundColor Green
    Write-Host '[V25f] Unknown, OOF, test, L3, full cascade, rejection, stable and P95 remain intentionally unexecuted.' -ForegroundColor Green
    switch ($Stage) {
        'prepare' { Invoke-Prepare }
        'smoke' { Invoke-Smoke }
        'analyze' { Invoke-Analyze }
        'all' { Invoke-Prepare; Invoke-Smoke; Invoke-Analyze }
    }
    Write-Host '[V25f] requested stage completed.' -ForegroundColor Green
}
catch { [Console]::Error.WriteLine('[V25f] pipeline stopped: ' + $_.Exception.Message); exit 1 }
