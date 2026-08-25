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
$contractPath = Join-Path $projectPath 'data\processed\RADAR-LN3-V25g-cq08-known-gate-causal-decision-policy-contract.json'
$readinessPath = Join-Path $projectPath 'src-v3\processed\report\RADAR-LN3-V25g-known-gate-causal-decision-policy-readiness.json'
$reportPath = Join-Path $projectPath 'src-v3\processed\report\RADAR-LN3-V25g-known-gate-causal-decision-policy.json'
$sourceArtifactPath = Join-Path $projectPath 'artifacts\RADAR-LN3-V25a-cq08-liquid-known-gate-course-preclip-gradient-telemetry-seed43-formal'
$trainFeatureCachePath = Join-Path $projectPath 'data\processed\RADAR-LN3-V21e-cq08-known-gate-train-causal-course-consistency-cache.npz'
$validationFeatureCachePath = Join-Path $projectPath 'data\processed\RADAR-LN3-V21e-cq08-known-gate-val-causal-course-consistency-cache.npz'
$maxFeatureCacheBytes = 64MB

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

function Assert-NoV25gTemporaryFiles {
    $temporaryPaths = @($contractPath + '.tmp', $readinessPath + '.tmp', $reportPath + '.tmp')
    $residual = @($temporaryPaths | Where-Object { Test-Path -LiteralPath $_ -PathType Leaf })
    if ($residual.Count -gt 0) {
        throw ('V25g residual temporary artifact(s) found: ' + ($residual -join '; '))
    }
    if (Test-Path -LiteralPath $sourceArtifactPath -PathType Container) {
        $sourceResidual = @(Get-ChildItem -LiteralPath $sourceArtifactPath -Recurse -Force -File | Where-Object { $_.Name -match '\.(tmp|partial|part|lock)$' })
        if ($sourceResidual.Count -gt 0) {
            throw ('V25a frozen artifact contains residual temporary file(s): ' + (($sourceResidual | ForEach-Object { $_.FullName }) -join '; '))
        }
    }
    foreach ($cachePath in @($trainFeatureCachePath, $validationFeatureCachePath)) {
        if (-not (Test-Path -LiteralPath $cachePath -PathType Leaf)) {
            throw ('V25a input feature cache is missing: ' + $cachePath)
        }
        $cacheInfo = Get-Item -LiteralPath $cachePath
        if ([int64]$cacheInfo.Length -gt [int64]$maxFeatureCacheBytes) {
            throw ('V25a input feature cache exceeds the 64 MiB envelope: ' + $cachePath)
        }
        foreach ($suffix in @('.tmp', '.partial', '.part', '.lock')) {
            $residualCachePath = $cachePath + $suffix
            if (Test-Path -LiteralPath $residualCachePath -PathType Leaf) {
                throw ('V25a input feature cache has a residual temporary file: ' + $residualCachePath)
            }
        }
    }
}

function Assert-Readiness {
    Assert-File $contractPath 'V25g contract'
    Assert-File $readinessPath 'V25g readiness'
    Assert-NoV25gTemporaryFiles
    $contract = Read-Utf8Json $contractPath
    $readiness = Read-Utf8Json $readinessPath
    if ([string]$contract.schema_version -ne 'RADAR-LN3-V25g-known-gate-causal-decision-policy-experiment-contract-v1' -or [string]$readiness.schema_version -ne 'RADAR-LN3-V25g-known-gate-causal-decision-policy-v1' -or [string]$readiness.status -ne 'passed' -or [string]$readiness.training -ne 'not_executed' -or [string]$readiness.calibration -ne 'train_only_frozen_checkpoint_in_sample' -or [bool]$readiness.validation_used_for_threshold_fit -ne $false -or [int]$readiness.test_payload_access_count -ne 0 -or [int]$readiness.unknown_payload_access_count -ne 0) {
        throw ('V25g readiness is stale or outside frozen action-policy scope: ' + $readinessPath)
    }
    foreach ($property in @($readiness.checks.PSObject.Properties)) {
        if ([bool]$property.Value -ne $true) { throw ('V25g readiness check failed: ' + $property.Name) }
    }
}

function Invoke-Prepare {
    Invoke-RadarPython @('-m', 'py_compile', 'radar_ln3\known_gate.py', 'radar_ln3\known_gate_v25g_policy.py', 'build\known_gate_v25f.py', 'build\known_gate_v25g.py', 'report\known_gate_v25f.py', 'report\known_gate_v25g.py', 'smoke\known_gate_v25g_contract_test.py')
    Invoke-RadarPython @('-m', 'build.known_gate_v25g', '--project-root', $projectPath)
    Assert-Readiness
}

function Invoke-Smoke {
    Assert-Readiness
    Invoke-RadarPython @('-m', 'smoke.known_gate_v25g_contract_test', '--project-root', $projectPath)
    Invoke-RadarPython @('-m', 'smoke.static_isolation_check', '--root', $srcRoot)
}

function Invoke-Analyze {
    Assert-Readiness
    Write-Host '[V25g] evaluating frozen V25a epoch14 on train calibration and validation policy paths' -ForegroundColor Green
    Write-Host '[V25g] train-only calibration is explicitly non-OOF; no deployment or formal adoption claim is allowed' -ForegroundColor Yellow
    Invoke-RadarPython @('-m', 'report.known_gate_v25g', '--project-root', $projectPath)
    Assert-File $reportPath 'V25g policy report'
    Assert-NoV25gTemporaryFiles
    $report = Read-Utf8Json $reportPath
    if ([string]$report.status -ne 'completed_development_policy_diagnostic' -or [string]$report.schema_version -ne 'RADAR-LN3-V25g-known-gate-causal-decision-policy-v1' -or [bool]$report.calibration.is_oof -ne $false -or [string]$report.OOF -ne 'not_executed' -or [string]$report.test -ne 'not_executed' -or [bool]$report.deployment_eligible) {
        throw ('V25g report violates the frozen action-policy scope: ' + $reportPath)
    }
}

try {
    Assert-File $python 'Project Python interpreter'
    Write-Host ('[V25g] project=' + $projectPath) -ForegroundColor Green
    Write-Host ('[V25g] stage=' + $Stage + '; frozen V25a epoch14 causal ACCEPT/PENDING policy diagnostic') -ForegroundColor Green
    Write-Host '[V25g] No training, checkpoint mutation, Unknown, test, exact OOF, L3, cascade, rejection, stable or P95.' -ForegroundColor Green
    switch ($Stage) {
        'prepare' { Invoke-Prepare }
        'smoke' { Invoke-Smoke }
        'analyze' { Invoke-Analyze }
        'all' { Invoke-Prepare; Invoke-Smoke; Invoke-Analyze }
    }
    Write-Host '[V25g] requested stage completed.' -ForegroundColor Green
}
catch { [Console]::Error.WriteLine('[V25g] pipeline stopped: ' + $_.Exception.Message); exit 1 }
