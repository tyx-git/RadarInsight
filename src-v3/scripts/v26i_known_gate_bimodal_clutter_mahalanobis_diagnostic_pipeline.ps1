[CmdletBinding()]
param(
    [ValidateSet('prepare', 'smoke', 'diagnose', 'all')]
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
$contractPath = Join-Path $projectPath 'data\processed\RADAR-LN3-V26i-cq08-l1-train-only-bimodal-clutter-mahalanobis-diagnostic-contract.json'
$readinessPath = Join-Path $projectPath 'src-v3\processed\report\RADAR-LN3-V26i-l1-train-only-bimodal-clutter-mahalanobis-diagnostic-readiness.json'
$reportPath = Join-Path $projectPath 'src-v3\processed\report\RADAR-LN3-V26i-l1-train-only-bimodal-clutter-mahalanobis-diagnostic.json'
$detailPath = Join-Path $projectPath 'src-v3\processed\report\RADAR-LN3-V26i-l1-train-only-bimodal-clutter-mahalanobis-diagnostic-parent-details.csv'
$maxReportBytes = 8 * 1024 * 1024
$maxDetailBytes = 3 * 1024 * 1024

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
    finally { Pop-Location }
    if ($code -ne 0) { throw ('Python command failed with exit code ' + $code + ': ' + $display) }
}

function Assert-File {
    param([Parameter(Mandatory = $true)][string]$Path, [Parameter(Mandatory = $true)][string]$Description)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { throw ($Description + ' is missing: ' + $Path) }
}

function Read-Utf8Json {
    param([Parameter(Mandatory = $true)][string]$Path)
    Assert-File -Path $Path -Description 'JSON artifact'
    return ConvertFrom-Json -InputObject ([System.IO.File]::ReadAllText($Path, [System.Text.UTF8Encoding]::new($false, $true)))
}

function Get-Sha256 {
    param([Parameter(Mandatory = $true)][string]$Path)
    # Avoid relying on Get-FileHash module auto-loading in a detached
    # Windows PowerShell process.  The raw SHA-256 value is the same while
    # remaining available in both Windows PowerShell 5.1 and PowerShell 7.
    $stream = [System.IO.File]::OpenRead($Path)
    $sha256 = [System.Security.Cryptography.SHA256]::Create()
    try {
        return ([System.BitConverter]::ToString($sha256.ComputeHash($stream))).Replace('-', '').ToLowerInvariant()
    }
    finally {
        $sha256.Dispose()
        $stream.Dispose()
    }
}

function Assert-Prepared {
    Assert-File -Path $contractPath -Description 'V26i contract'
    Assert-File -Path $readinessPath -Description 'V26i readiness'
    $contract = Read-Utf8Json -Path $contractPath
    $readiness = Read-Utf8Json -Path $readinessPath
    if ([string]$contract.schema_version -ne 'RADAR-LN3-V26i-l1-train-only-bimodal-clutter-mahalanobis-diagnostic-contract-v1' -or
        [string]$readiness.schema_version -ne 'RADAR-LN3-V26i-l1-train-only-bimodal-clutter-mahalanobis-diagnostic-readiness-v1' -or
        [string]$readiness.status -ne 'passed' -or
        [string]$readiness.contract_sha256 -ne (Get-Sha256 -Path $contractPath) -or
        [int]$contract.fold_count -ne 5 -or
        [int]$contract.unknown_reference_fit_exposure -ne 0 -or
        [int]$contract.unknown_threshold_fit_exposure -ne 0 -or
        [bool]$contract.new_training -or
        [bool]$contract.new_threshold_fit -or
        [bool]$contract.semantic_action_mutation -or
        [bool]$contract.deployment_eligible -or
        [int]$contract.ood_score_policy.known_clutter_reference.component_count_fixed -ne 2 -or
        [double]$contract.future_action_gate.known_target_recall_minimum -ne 0.95) {
        throw 'V26i preparation artifacts violate the frozen train-only fixed-K=2 diagnostic contract.'
    }
    $checks = @($readiness.checks.PSObject.Properties)
    if ($checks.Count -eq 0 -or @($checks | Where-Object { $_.Value -ne $true }).Count -ne 0) {
        throw 'V26i readiness has a failed or missing check.'
    }
}

function Invoke-Prepare {
    Invoke-RadarPython @('-m', 'py_compile',
        'radar_ln3\known_gate_v26i.py',
        'build\known_gate_v26i.py',
        'report\known_gate_v26i.py',
        'smoke\known_gate_v26i_contract_test.py')
    $preparationIsCurrent = $false
    if ((Test-Path -LiteralPath $contractPath -PathType Leaf) -and (Test-Path -LiteralPath $readinessPath -PathType Leaf)) {
        # ``srcRoot`` is the fixed ``<project>/src-v3`` directory.  Deriving
        # the root here avoids embedded quotes in a Python ``-c`` argument,
        # which Windows PowerShell 5.1 strips while serialising native-process
        # arguments.
        $validationProgram = 'from pathlib import Path; from radar_ln3.known_gate_v26i import validate_prepared_contract; validate_prepared_contract(Path.cwd().parent)'
        Push-Location -LiteralPath $srcRoot
        try {
            try {
                & $python -X utf8 -c $validationProgram *> $null
                $preparationIsCurrent = ([int]$LASTEXITCODE -eq 0)
            }
            catch {
                # A stale immutable binding is expected after implementation
                # changes.  It is handled below by archival and regeneration;
                # it must not escape as a PowerShell pipeline error.
                $preparationIsCurrent = $false
            }
        }
        finally {
            Pop-Location
        }
    }
    if (-not $preparationIsCurrent -and ((Test-Path -LiteralPath $contractPath) -or (Test-Path -LiteralPath $readinessPath))) {
        $archiveStamp = Get-Date -Format 'yyyyMMdd-HHmmss'
        foreach ($artifactPath in @($contractPath, $readinessPath)) {
            if (Test-Path -LiteralPath $artifactPath -PathType Leaf) {
                $archivePath = $artifactPath + '.stale-before-script-runtime-fix-' + $archiveStamp
                Move-Item -LiteralPath $artifactPath -Destination $archivePath -ErrorAction Stop
                Write-Host ('[V26i] archived stale preparation artifact to ' + $archivePath) -ForegroundColor Yellow
            }
        }
    }
    Invoke-RadarPython @('-m', 'build.known_gate_v26i', '--project-root', $projectPath)
    Assert-Prepared
}

function Invoke-Smoke {
    Assert-Prepared
    Invoke-RadarPython @('-m', 'smoke.known_gate_v26i_contract_test', '--project-root', $projectPath)
}

function Invoke-Diagnose {
    Assert-Prepared
    Invoke-RadarPython @('-m', 'report.known_gate_v26i', '--project-root', $projectPath)
    Assert-File -Path $reportPath -Description 'V26i ranking report'
    $report = Read-Utf8Json -Path $reportPath
    if (([string]$report.status -notin @('completed_v26i_train_only_bimodal_clutter_mahalanobis_ranking_diagnostic', 'completed_v26i_infeasible_fixed_k2_reference')) -or
        [string]$report.schema_version -ne 'RADAR-LN3-V26i-l1-train-only-bimodal-clutter-mahalanobis-diagnostic-v1' -or
        [bool]$report.deployment_eligible) {
        throw 'V26i report violates the frozen train-only diagnostic scope.'
    }
    if ((Get-Item -LiteralPath $reportPath).Length -gt $maxReportBytes) { throw ('V26i report exceeds bounded size: ' + $reportPath) }
    if ([string]$report.status -eq 'completed_v26i_train_only_bimodal_clutter_mahalanobis_ranking_diagnostic') {
        Assert-File -Path $detailPath -Description 'V26i parent detail CSV'
        if ((Get-Item -LiteralPath $detailPath).Length -gt $maxDetailBytes) { throw ('V26i parent detail exceeds bounded size: ' + $detailPath) }
        if ([int]$report.future_action_gate.known_target_recall_minimum -ne 0.95 -or
            [string]$report.future_action_gate.status -ne 'not_evaluable_v26i_does_not_change_or_replay_actions') {
            throw 'V26i report lost the KnownTarget Recall >= 0.95 future-action gate boundary.'
        }
    }
}

try {
    Assert-File -Path $python -Description 'Project Python interpreter'
    Write-Host ('[V26i] project=' + $projectPath) -ForegroundColor Green
    Write-Host ('[V26i] stage=' + $Stage + '; V26h control with only train-Known Clutter reference geometry changed from one to fixed two centres') -ForegroundColor Green
    Write-Host '[V26i] No training, cache rebuild, threshold fitting, action replay, L3 mutation or deployment registry is executed. KnownTarget Recall >= 0.95 is recorded only for any future action experiment.' -ForegroundColor Green
    switch ($Stage) {
        'prepare' { Invoke-Prepare }
        'smoke' { Invoke-Smoke }
        'diagnose' { Invoke-Diagnose }
        'all' { Invoke-Prepare; Invoke-Smoke; Invoke-Diagnose }
    }
    Write-Host '[V26i] requested stage completed.' -ForegroundColor Green
}
catch {
    [Console]::Error.WriteLine('[V26i] pipeline stopped: ' + $_.Exception.Message)
    exit 1
}
