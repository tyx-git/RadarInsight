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
$defaultProjectRoot = (Resolve-Path -LiteralPath (Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $srcRoot)))).Path
$projectPath = [System.IO.Path]::GetFullPath($(if ([string]::IsNullOrWhiteSpace($ProjectRoot)) { $defaultProjectRoot } else { $ProjectRoot }))
$python = Join-Path $projectPath '.venv\Scripts\python.exe'
$readinessPath = Join-Path $projectPath 'src-v3\processed\report\RADAR-LN3-V24d-known-gate-l3-oof-score-band-pending-calibration-readiness.json'
$reportPath = Join-Path $projectPath 'src-v3\processed\report\RADAR-LN3-V24d-known-gate-l3-oof-score-band-pending-cross-fitted.json'

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
    finally { Pop-Location }
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

function Assert-Readiness {
    Assert-File $readinessPath 'V24d calibration readiness'
    $readiness = Read-Utf8Json $readinessPath
    $issues = [System.Collections.Generic.List[string]]::new()
    if ([string]$readiness.status -ne 'passed') { $issues.Add('status') }
    if ([string]$readiness.schema_version -ne 'RADAR-LN3-V24d-known-gate-l3-oof-score-band-readiness-v1') { $issues.Add('schema_version') }
    if ([int]$readiness.known_train_parent_count -ne 986 -or [int]$readiness.clutter_parent_support -ne 33 -or [int]$readiness.fold_count -ne 5) { $issues.Add('support') }
    if ([string]$readiness.threshold_fit_unit -ne 'parent_final_OOF_only' -or [bool]$readiness.threshold_fit_held_out_labels_accessed) { $issues.Add('threshold_fit_scope') }
    if ([int]$readiness.validation_payload_access_count -ne 0 -or [int]$readiness.test_payload_access_count -ne 0 -or [int]$readiness.unknown_training_exposure -ne 0) { $issues.Add('sealed_scope') }
    if ([string]$readiness.formal_training -ne 'not_executed_not_applicable' -or [bool]$readiness.deployment_eligible) { $issues.Add('training_or_deployment') }
    foreach ($property in @($readiness.checks.PSObject.Properties)) {
        if ([bool]$property.Value -ne $true) { $issues.Add('check:' + $property.Name) }
    }
    if ($issues.Count -ne 0) {
        throw ('V24d readiness is stale or failed: ' + ($issues -join ',') + '; path=' + $readinessPath)
    }
}

function Invoke-Prepare {
    Invoke-RadarPython @('-m', 'py_compile', 'build\known_gate_v24d.py', 'report\known_gate_v24d.py')
    Invoke-RadarPython @('-m', 'build.known_gate_v24d', '--project-root', $projectPath)
    Assert-Readiness
}


function Invoke-Evaluate {
    Assert-Readiness
    Invoke-RadarPython @('-m', 'report.known_gate_v24d', '--project-root', $projectPath)
    Assert-File $reportPath 'V24d cross-fitted OOF report'
    $report = Read-Utf8Json $reportPath
    $allowedStatuses = @('completed_cross_fitted_oof_calibration_candidate', 'completed_cross_fitted_oof_gate_failed')
    if ($allowedStatuses -notcontains [string]$report.status) { throw ('V24d report status is invalid: ' + $report.status) }
    if ([string]$report.schema_version -ne 'RADAR-LN3-V24d-known-gate-l3-oof-score-band-report-v1') { throw 'V24d report schema differs from the frozen contract.' }
    if ([bool]$report.deployment_eligible -or [string]$report.test -ne 'not_executed' -or [string]$report.validation -ne 'not_executed') { throw 'V24d report violates the sealed OOF-only scope.' }
    if ($null -eq $report.cross_fit_protocol.fold_thresholds -or @($report.cross_fit_protocol.fold_thresholds).Count -ne 5 -or $null -eq $report.parent_final_wilson_intervals -or $null -eq $report.paired_component_bootstrap -or $null -eq $report.parent_final_action_changes -or $null -eq $report.provenance_audit) { throw 'V24d report lacks required OOF calibration evidence.' }
    $candidateSpace = $report.cross_fit_protocol.candidate_space
    if ($null -eq $candidateSpace -or [string]$candidateSpace.method -ne 'all score decision-equivalence breakpoints' -or [bool]$candidateSpace.uses_approximation_grid) { throw 'V24d report is not bound to the exact breakpoint search.' }
    if ([bool]$report.cross_fit_protocol.held_out_fold_labels_used_for_threshold_selection -or [bool]$report.cross_fit_protocol.prefix_or_window_labels_used_for_threshold_selection) { throw 'V24d report violates threshold-fit isolation.' }
    $frozen = $report.frozen_decision
    if ($null -eq $frozen -or $frozen.no_l1_model_change_or_retraining -ne $true -or $frozen.pending_is_not_unknown_rejection -ne $true -or $frozen.development_or_oof_only_no_release_claim -ne $true) { throw 'V24d report lacks sealed L3-only decision semantics.' }
    foreach ($fold in @($report.cross_fit_protocol.fold_thresholds)) {
        if ([int]$fold.candidate_count -le 0 -or [string]$fold.candidate_space -notlike '*breakpoint*' -or [int]$fold.held_out_fold_id -notin 0..4) { throw 'V24d report has an invalid cross-fitted threshold record.' }
    }
}

try {
    Assert-File $python 'Project Python interpreter'
    Write-Host ('[V24d] project=' + $projectPath) -ForegroundColor Green
    Write-Host ('[V24d] stage=' + $Stage + '; OOF-only L3 score-band Pending calibration, no L1 training') -ForegroundColor Green
    Write-Host '[V24d] Existing V21e treatment OOF scores are hash-bound. Validation, test, Unknown, raw payloads, L1 checkpoints, full cascade, rejection, stable and P95 remain unexecuted.' -ForegroundColor Green
    switch ($Stage) {
        'prepare' { Invoke-Prepare }
        'evaluate' { Invoke-Evaluate }
        'all' { Invoke-Prepare; Invoke-Evaluate }
    }
    Write-Host '[V24d] requested stage completed.' -ForegroundColor Green
}
catch { [Console]::Error.WriteLine('[V24d] pipeline stopped: ' + $_.Exception.Message); exit 1 }
