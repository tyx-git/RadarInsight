[CmdletBinding()]
param(
    [ValidateSet('control', 'treatment', 'both')]
    [string]$Arm = 'both',
    [ValidateSet('formal', 'oof', 'calibrate', 'all')]
    [string]$Stage = 'all',
    [ValidateRange(0, 4)]
    [int[]]$OuterFold = @(0, 1, 2, 3, 4),
    [int]$PredictionBatchSize = 2,
    [switch]$PreflightOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$srcRoot = Split-Path -Parent $scriptRoot
$projectRoot = Split-Path -Parent $srcRoot
$python = Join-Path $projectRoot '.venv\Scripts\python.exe'

$taskSlugs = [ordered]@{
    dbb_cu_gate = 'dbb-cu-gate'
    dbb_specialist = 'dbb-specialist'
    cu_specialist = 'cu-specialist'
}

function Get-ArmList {
    if ($Arm -eq 'both') { return @('control', 'treatment') }
    return @($Arm)
}

function Get-ConfigPath([string]$currentArm, [string]$taskName) {
    $slug = $taskSlugs[$taskName]
    return Join-Path $projectRoot (
        'config\RADAR-LN3\RADAR-LN3-V15-cq08-liquid-strict-open-set-angular-knownness-' +
        $currentArm + '-' + $slug + '-config.json'
    )
}

function Get-FormalRoot([string]$currentArm) {
    return Join-Path $projectRoot (
        'artifacts\RADAR-LN3-V15-cq08-liquid-strict-open-set-angular-knownness-' +
        $currentArm + '-formal'
    )
}

function Get-NestedRoot([string]$currentArm, [int]$foldId) {
    return Join-Path $projectRoot (
        'artifacts\RADAR-LN3-V15-cq08-liquid-strict-open-set-angular-knownness-' +
        $currentArm + '-nested-oof\outer-' + $foldId.ToString('00')
    )
}

function Invoke-RadarPython([string[]]$Arguments) {
    Write-Host ('> ' + $python + ' ' + ($Arguments -join ' ')) -ForegroundColor Cyan
    Push-Location $srcRoot
    try {
        & $python @Arguments
    }
    finally {
        Pop-Location
    }
    if ($LASTEXITCODE -ne 0) {
        throw ('Python command failed with exit code ' + $LASTEXITCODE)
    }
}

function Assert-Path([string]$path, [string]$description) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw ($description + ' is missing: ' + $path)
    }
}

function Assert-FormalCheckpointSet([string]$currentArm) {
    $formalRoot = Get-FormalRoot $currentArm
    $summaryPath = Join-Path $formalRoot 'training-summary.json'
    Assert-Path $summaryPath ('V15 ' + $currentArm + ' formal training summary')
    # Windows PowerShell 5.1 otherwise decodes the UTF-8, no-BOM training summary
    # through the active ANSI code page; Chinese status strings then corrupt JSON parsing.
    $summary = Get-Content -LiteralPath $summaryPath -Raw -Encoding UTF8 | ConvertFrom-Json
    foreach ($fieldName in @('status', 'schema_version', 'unknown_training_exposure', 'development_unknown_access_count_during_formal_training')) {
        if (-not ($summary.PSObject.Properties.Name -contains $fieldName)) {
            throw ('V15 ' + $currentArm + " formal training summary is missing required field ${fieldName}: " + $summaryPath)
        }
    }
    if ($summary.status -ne 'passed' -or $summary.schema_version -ne 'RADAR-LN3-V15') {
        throw ('V15 ' + $currentArm + ' formal training summary is not a completed V15 run: ' + $summaryPath)
    }
    if ([int]$summary.unknown_training_exposure -ne 0 -or [int]$summary.development_unknown_access_count_during_formal_training -ne 0) {
        throw ('V15 ' + $currentArm + ' formal training violated the strict open-set zero-exposure contract: ' + $summaryPath)
    }
    foreach ($taskName in $taskSlugs.Keys) {
        Assert-Path (Join-Path $formalRoot ($taskName + '\best_model.pt')) (
            'V15 ' + $currentArm + ' formal checkpoint'
        )
    }
    Assert-Path (Join-Path $formalRoot 'joint_best_model.pt') ('V15 ' + $currentArm + ' formal joint checkpoint')
}

function Get-StageResume([string]$stageRoot) {
    $summary = Join-Path $stageRoot 'stage-summary.json'
    if (Test-Path -LiteralPath $summary -PathType Leaf) {
        return $null
    }
    $resume = Join-Path $stageRoot 'joint_last_model.pt'
    if (Test-Path -LiteralPath $resume -PathType Leaf) {
        return $resume
    }
    if (Test-Path -LiteralPath $stageRoot) {
        $items = @(Get-ChildItem -LiteralPath $stageRoot -Force)
        if ($items.Count -gt 0) {
            throw ('Nested OOF stage is non-empty without stage-summary.json or joint_last_model.pt: ' + $stageRoot)
        }
    }
    return $null
}

function Invoke-Formal([string]$currentArm) {
    $formalRoot = Get-FormalRoot $currentArm
    if (Test-Path -LiteralPath $formalRoot) {
        $items = @(Get-ChildItem -LiteralPath $formalRoot -Force)
        if ($items.Count -gt 0) {
            throw ('Refusing to overwrite non-empty V15 formal artifact. Use the explicit resume workflow only after manual audit: ' + $formalRoot)
        }
    }
    Invoke-RadarPython @(
        '-m', 'train.cascade',
        '--project-root', $projectRoot,
        '--gate-config', (Get-ConfigPath $currentArm 'dbb_cu_gate'),
        '--dbb-config', (Get-ConfigPath $currentArm 'dbb_specialist'),
        '--cu-config', (Get-ConfigPath $currentArm 'cu_specialist')
    )
    Assert-FormalCheckpointSet $currentArm
}

function Invoke-NestedOOF([string]$currentArm, [int]$foldId) {
    $root = Get-NestedRoot $currentArm $foldId
    $prediction = Join-Path $root 'outer-held-out-predictions.json'
    $claim = Join-Path $root 'outer-held-out-forward-claim.json'
    if (Test-Path -LiteralPath $prediction -PathType Leaf) {
        Assert-Path $claim ('V15 ' + $currentArm + ' completed outer-held-out forward claim')
        Write-Host ('Nested OOF ' + $currentArm + ' fold ' + $foldId.ToString('00') + ' already completed; preserving it.') -ForegroundColor Yellow
        return
    }
    if (Test-Path -LiteralPath $claim -PathType Leaf) {
        throw ('Outer held-out forward was previously claimed without a completed prediction. Do not retry automatically; inspect manually: ' + $claim)
    }
    $innerRoot = Join-Path $root 'inner-selection'
    $outerRoot = Join-Path $root 'outer-retrain'
    $innerResume = Get-StageResume $innerRoot
    $outerResume = Get-StageResume $outerRoot
    $arguments = @(
        '-m', 'train.angular_knownness_oof',
        '--project-root', $projectRoot,
        '--arm', $currentArm,
        '--outer-fold-id', $foldId.ToString(),
        '--prediction-batch-size', $PredictionBatchSize.ToString()
    )
    if ($null -ne $innerResume) {
        $arguments += @('--inner-resume-checkpoint', $innerResume)
    }
    if ($null -ne $outerResume) {
        $arguments += @('--outer-resume-checkpoint', $outerResume)
    }
    Invoke-RadarPython $arguments
    Assert-Path $prediction ('V15 ' + $currentArm + ' nested OOF prediction')
    Assert-Path $claim ('V15 ' + $currentArm + ' completed outer-held-out forward claim')
}

function Invoke-Calibration([string]$currentArm) {
    Assert-FormalCheckpointSet $currentArm
    Invoke-RadarPython @(
        '-m', 'report.angular_knownness_oof',
        '--project-root', $projectRoot,
        '--arm', $currentArm
    )
}

function Invoke-Preflight {
    Assert-Path $python 'Python environment'
    $requiresExistingFormal = $Stage -in @('oof', 'calibrate')
    if ($requiresExistingFormal) {
        foreach ($currentArm in (Get-ArmList)) {
            Assert-FormalCheckpointSet $currentArm
        }
        Invoke-RadarPython @('-m', 'build.angular_knownness', '--project-root', $projectRoot, '--verify-only')
    }
    else {
        Invoke-RadarPython @('-m', 'build.angular_knownness', '--project-root', $projectRoot)
    }
    Invoke-RadarPython @('-m', 'build.angular_knownness_oof', '--project-root', $projectRoot)
    Invoke-RadarPython @('-m', 'smoke.angular_knownness_contract_test', '--project-root', $projectRoot)
    Invoke-RadarPython @('-m', 'smoke.static_isolation_check', '--root', $srcRoot)
}

try {
    Write-Host ('[V15] project=' + $projectRoot) -ForegroundColor Green
    Write-Host ('[V15] stage=' + $Stage + '; arm=' + $Arm + '; folds=' + ($OuterFold -join ',')) -ForegroundColor Green
    Invoke-Preflight
    if ($PreflightOnly) {
        Write-Host '[V15] preflight passed; no formal training, OOF, calibration, development audit, test, P95, or deployment was executed.' -ForegroundColor Green
        exit 0
    }
    foreach ($currentArm in (Get-ArmList)) {
        if ($Stage -in @('formal', 'all')) {
            Invoke-Formal $currentArm
        }
        if ($Stage -in @('oof', 'all')) {
            foreach ($foldId in $OuterFold) {
                Invoke-NestedOOF $currentArm $foldId
            }
        }
        if ($Stage -in @('calibrate', 'all')) {
            Invoke-Calibration $currentArm
        }
    }
    Write-Host '[V15] completed requested stages. Development Unknown audit, test, P95, and deployment remain unexecuted.' -ForegroundColor Green
}
catch {
    Write-Error ('V15 angular-knownness pipeline stopped: ' + $_.Exception.Message)
    exit 1
}
