[CmdletBinding()]
param(
    [ValidateSet('prepare', 'evaluate', 'all')]
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
$reportPath = Join-Path $projectPath 'src-v3\processed\report\RADAR-LN3-V23e-known-gate-parent-balanced-checkpoint-selection.json'

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
    param([Parameter(Mandatory = $true)][string]$Path, [Parameter(Mandatory = $true)][string]$Description)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw ($Description + ' is missing: ' + $Path)
    }
}

function Read-Utf8Json {
    param([Parameter(Mandatory = $true)][string]$Path)
    return ConvertFrom-Json -InputObject ([System.IO.File]::ReadAllText($Path, [System.Text.UTF8Encoding]::new($false, $true)))
}

function Assert-V23eReport {
    Assert-File $reportPath 'V23e selector report'
    $report = Read-Utf8Json $reportPath
    $issues = [System.Collections.Generic.List[string]]::new()
    if ([string]$report.status -ne 'completed_development_selector_only') { $issues.Add('status') }
    if ([string]$report.schema_version -ne 'RADAR-LN3-V23e-known-gate-parent-balanced-checkpoint-selection-report-v1') { $issues.Add('schema_version') }
    if ($report.scanned_all_frozen_epochs -ne $true) { $issues.Add('scanned_all_frozen_epochs') }
    if (@($report.epoch_candidates).Count -ne 20) { $issues.Add('epoch_candidates') }
    if ([int]$report.selector_protocol.clutter_parent_support -ne 9) { $issues.Add('clutter_support') }
    if ([int]$report.selector_protocol.parent_final_clutter_true_positive_count_minimum -ne 8) { $issues.Add('clutter_true_positive_count_minimum') }
    if ([double]$report.selector_protocol.parent_final_clutter_precision_minimum -ne 0.90) { $issues.Add('clutter_precision_minimum') }
    if ($report.deployment_eligible -ne $false) { $issues.Add('deployment_eligible') }
    if ($issues.Count -ne 0) { throw ('V23e report violates the frozen selector contract: ' + ($issues -join ',') + '; path=' + $reportPath) }
}

function Invoke-Prepare {
    Invoke-RadarPython @('-m', 'py_compile', 'radar_ln3\known_gate.py', 'radar_ln3\known_gate_v23e_selection.py', 'report\known_gate_v23e_selection.py')
}


function Invoke-Evaluate {
    # This is a read-only replay of saved V23a epoch checkpoints.  No optimizer
    # step, new checkpoint, OOF, test, Unknown or product evaluation is allowed.
    foreach ($range in @(@(1, 5), @(6, 10), @(11, 15), @(16, 20))) {
        $start = [int]$range[0]
        $end = [int]$range[1]
        $name = 'src-v3/processed/report/RADAR-LN3-V23e-known-gate-parent-balanced-checkpoint-selection-part-' + $start.ToString('00') + '-' + $end.ToString('00') + '.json'
        Invoke-RadarPython @('-m', 'report.known_gate_v23e_selection', '--project-root', $projectPath, '--start-epoch', $start, '--end-epoch', $end, '--output', $name)
    }
    Invoke-RadarPython @('-m', 'report.known_gate_v23e_selection', '--project-root', $projectPath, '--merge-parts')
    Assert-V23eReport
}

try {
    Assert-File $python 'Project Python interpreter'
    Write-Host ('[V23e] project=' + $projectPath) -ForegroundColor Green
    Write-Host ('[V23e] stage=' + $Stage + '; read-only checkpoint-selector diagnostic; no training') -ForegroundColor Green
    switch ($Stage) {
        'prepare' { Invoke-Prepare }
        'evaluate' { Invoke-Evaluate }
        'all' { Invoke-Prepare; Invoke-Evaluate }
    }
    Write-Host '[V23e] requested stage completed.' -ForegroundColor Green
}
catch {
    [Console]::Error.WriteLine('[V23e] pipeline stopped: ' + $_.Exception.Message)
    exit 1
}
