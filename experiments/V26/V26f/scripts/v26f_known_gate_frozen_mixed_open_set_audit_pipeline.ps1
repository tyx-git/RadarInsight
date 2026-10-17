[CmdletBinding()]
param(
    [ValidateSet('prepare', 'audit', 'all')]
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
$contractPath = Join-Path $projectPath 'data\processed\RADAR-LN3-V26f-cq08-frozen-mixed-open-set-audit-contract.json'
$readinessPath = Join-Path $projectPath 'src-v3\processed\report\RADAR-LN3-V26f-frozen-mixed-open-set-audit-readiness.json'
$reportPath = Join-Path $projectPath 'src-v3\processed\report\RADAR-LN3-V26f-frozen-mixed-open-set-audit.json'
$detailPath = Join-Path $projectPath 'src-v3\processed\report\RADAR-LN3-V26f-frozen-mixed-open-set-audit-parent-details.csv'
$maxReportBytes = 8 * 1024 * 1024
$maxDetailBytes = 4 * 1024 * 1024

function Invoke-RadarPython {
    param([Parameter(Mandatory = $true)][string[]]$Arguments)
    $runtimeArguments = @('-X', 'utf8') + $Arguments
    $display = $python + ' ' + ($runtimeArguments -join ' ')
    Write-Host ('> ' + $display) -ForegroundColor Cyan
    $code = 1
    Push-Location -LiteralPath $srcRoot
    try {
        & $python @runtimeArguments | Out-Host
        $code = [int]$LASTEXITCODE
    }
    finally {
        Pop-Location
    }
    if ($code -ne 0) {
        throw ('Python command failed with exit code ' + $code + ': ' + $display)
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
    Assert-File -Path $Path -Description 'JSON artifact'
    return ConvertFrom-Json -InputObject ([System.IO.File]::ReadAllText($Path, [System.Text.UTF8Encoding]::new($false, $true)))
}

function Get-Sha256 {
    param([Parameter(Mandatory = $true)][string]$Path)
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Assert-Prepared {
    Assert-File -Path $contractPath -Description 'V26f contract'
    Assert-File -Path $readinessPath -Description 'V26f readiness'
    $contract = Read-Utf8Json -Path $contractPath
    $readiness = Read-Utf8Json -Path $readinessPath
    if ([string]$contract.schema_version -ne 'RADAR-LN3-V26f-frozen-mixed-open-set-audit-contract-v1' -or
        [string]$readiness.schema_version -ne 'RADAR-LN3-V26f-frozen-mixed-open-set-audit-readiness-v1' -or
        [string]$readiness.status -ne 'passed' -or
        [string]$readiness.contract_sha256 -ne (Get-Sha256 -Path $contractPath) -or
        [int]$contract.fold_count -ne 5 -or
        [int]$contract.unknown_training_exposure -ne 0 -or
        [int]$contract.unknown_normalization_exposure -ne 0 -or
        [int]$contract.unknown_threshold_fit_exposure -ne 0 -or
        [int]$contract.validation_access_count -ne 0 -or
        [int]$contract.test_access_count -ne 0 -or
        [bool]$contract.deployment_eligible) {
        throw 'V26f preparation artifacts violate the frozen inference-only contract.'
    }
    $checks = @($readiness.checks.PSObject.Properties)
    if ($checks.Count -eq 0 -or @($checks | Where-Object { $_.Value -ne $true }).Count -ne 0) {
        throw 'V26f readiness has a failed or missing check.'
    }
}

function Invoke-Prepare {
    Invoke-RadarPython @(
        '-m', 'py_compile',
        'radar_ln3\known_gate_v26f.py',
        'build\known_gate_v26f.py',
        'report\known_gate_v26f.py'
    )
    Invoke-RadarPython @('-m', 'build.known_gate_v26f', '--project-root', $projectPath)
    Assert-Prepared
}


function Invoke-Audit {
    Assert-Prepared
    Invoke-RadarPython @('-m', 'report.known_gate_v26f', '--project-root', $projectPath)
    Assert-File -Path $reportPath -Description 'V26f mixed audit report'
    Assert-File -Path $detailPath -Description 'V26f parent detail CSV'
    if ((Get-Item -LiteralPath $reportPath).Length -gt $maxReportBytes) {
        throw ('V26f report exceeds bounded size: ' + $reportPath)
    }
    if ((Get-Item -LiteralPath $detailPath).Length -gt $maxDetailBytes) {
        throw ('V26f parent detail CSV exceeds bounded size: ' + $detailPath)
    }
    $report = Read-Utf8Json -Path $reportPath
    if ([string]$report.status -ne 'completed_v26f_frozen_mixed_open_set_audit' -or
        [string]$report.schema_version -ne 'RADAR-LN3-V26f-frozen-mixed-open-set-audit-v1' -or
        [bool]$report.deployment_eligible -or
        [int]$report.isolation.unknown_training_exposure -ne 0 -or
        [int]$report.isolation.unknown_threshold_fit_exposure -ne 0 -or
        [bool]$report.routing_semantics.pending_is_not_unknown_rejection_success -ne $true) {
        throw 'V26f report violates its frozen mixed audit scope.'
    }
}

try {
    Assert-File -Path $python -Description 'Project Python interpreter'
    Write-Host ('[V26f] project=' + $projectPath) -ForegroundColor Green
    Write-Host ('[V26f] stage=' + $Stage + '; frozen V26e L1 epoch-32 + V26c L2 epoch-15 mixed Known/Unknown audit') -ForegroundColor Green
    Write-Host '[V26f] No training, OOF retraining, threshold fitting, or deployment registry is executed.' -ForegroundColor Green
    switch ($Stage) {
        'prepare' { Invoke-Prepare }
        'audit' { Invoke-Audit }
        'all' {
            Invoke-Prepare
            Invoke-Audit
        }
    }
    Write-Host '[V26f] requested stage completed.' -ForegroundColor Green
}
catch {
    [Console]::Error.WriteLine('[V26f] pipeline stopped: ' + $_.Exception.Message)
    exit 1
}
