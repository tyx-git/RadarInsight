[CmdletBinding()]
param(
    [ValidateSet('prepare', 'treatment', 'evaluate', 'all')]
    [string]$Stage = 'all',
    [ValidateRange(0, 4)]
    [int[]]$Folds = @(0, 1, 2, 3, 4),
    [ValidateRange(0, 4)]
    [int[]]$RestartPreCheckpointFolds = @(),
    [string]$ProjectRoot = ''
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$utf8NoBom = [System.Text.UTF8Encoding]::new($false)
[Console]::InputEncoding = $utf8NoBom
[Console]::OutputEncoding = $utf8NoBom
$OutputEncoding = $utf8NoBom

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$srcRoot = Split-Path -Parent $scriptRoot
$defaultProjectRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $srcRoot))
$projectPath = [System.IO.Path]::GetFullPath(
    $(if ([string]::IsNullOrWhiteSpace($ProjectRoot)) { $defaultProjectRoot } else { $ProjectRoot })
)
$python = Join-Path $projectPath '.venv\Scripts\python.exe'
$readinessPath = Join-Path $projectPath (
    'src-v3\processed\report\' +
    'RADAR-LN3-V22-known-gate-fixed-parent-balanced-oof-training-readiness.json'
)
$reportPath = Join-Path $projectPath (
    'src-v3\processed\report\' +
    'RADAR-LN3-V22-known-gate-fixed-parent-balanced-paired-oof.json'
)
$readinessSchema = 'RADAR-LN3-V22-known-gate-fixed-parent-balanced-oof-readiness-v1'
$predictionSchema = 'RADAR-LN3-V22-known-gate-fixed-parent-balanced-oof-prediction-v1'
$reportSchema = 'RADAR-LN3-V22-known-gate-fixed-parent-balanced-paired-oof-report-v1'
$allFolds = @(0, 1, 2, 3, 4)
$minimumFreeGpuMemoryMiB = 4096

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw ('Project Python interpreter is missing: ' + $python)
}

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

function Read-Utf8Json {
    param([Parameter(Mandatory = $true)][string]$Path)
    $text = [System.IO.File]::ReadAllText(
        $Path,
        [System.Text.UTF8Encoding]::new($false, $true)
    )
    return ConvertFrom-Json -InputObject $text
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

function Test-NonEmptyDirectory {
    param([Parameter(Mandatory = $true)][string]$Path)
    if (-not (Test-Path -LiteralPath $Path -PathType Container)) {
        return $false
    }
    return $null -ne (Get-ChildItem -LiteralPath $Path -Force | Select-Object -First 1)
}

function Get-ArtifactPath {
    param([Parameter(Mandatory = $true)][int]$Fold)
    return Join-Path $projectPath (
        'artifacts\RADAR-LN3-V22-cq08-known-gate-fixed-parent-balanced-oof\' +
        'fold-' + $Fold.ToString('00') + '\treatment'
    )
}

function Get-GpuMemoryState {
    $nvidiaSmiCommand = Get-Command 'nvidia-smi.exe' -ErrorAction SilentlyContinue
    if ($null -eq $nvidiaSmiCommand) {
        throw 'nvidia-smi.exe is unavailable; V22 cannot audit CUDA memory headroom.'
    }
    $queryOutput = @(
        & $nvidiaSmiCommand.Source @(
            '--id=0',
            '--query-gpu=memory.free,memory.total',
            '--format=csv,noheader,nounits'
        )
    )
    $queryExitCode = [int]$LASTEXITCODE
    if ($queryExitCode -ne 0 -or $queryOutput.Count -ne 1) {
        throw ('Unable to query GPU 0 memory with nvidia-smi; exit code=' + $queryExitCode)
    }
    $parts = @([string]$queryOutput[0] -split ',')
    if ($parts.Count -ne 2) {
        throw ('Unexpected nvidia-smi memory response: ' + [string]$queryOutput[0])
    }
    try {
        $freeMiB = [int]::Parse(
            $parts[0].Trim(),
            [System.Globalization.CultureInfo]::InvariantCulture
        )
        $totalMiB = [int]::Parse(
            $parts[1].Trim(),
            [System.Globalization.CultureInfo]::InvariantCulture
        )
    }
    catch {
        throw ('Unable to parse nvidia-smi memory response: ' + [string]$queryOutput[0])
    }
    return [pscustomobject]@{
        FreeMiB = $freeMiB
        TotalMiB = $totalMiB
    }
}

function Assert-GpuMemoryHeadroom {
    param([Parameter(Mandatory = $true)][int]$Fold)
    $memory = Get-GpuMemoryState
    Write-Host (
        '[V22] fold ' + $Fold + ' GPU 0 memory: free=' + $memory.FreeMiB +
        ' MiB; total=' + $memory.TotalMiB + ' MiB; required_free=' +
        $minimumFreeGpuMemoryMiB + ' MiB'
    ) -ForegroundColor Cyan
    if ([int]$memory.FreeMiB -lt $minimumFreeGpuMemoryMiB) {
        throw (
            'V22 fold ' + $Fold + ' CUDA preflight blocked: GPU 0 has only ' +
            $memory.FreeMiB + ' MiB free, below the required ' +
            $minimumFreeGpuMemoryMiB + ' MiB. Close GPU-heavy applications ' +
            '(for example games or overlays), then rerun the same command.'
        )
    }
}

function Move-PreCheckpointFoldArtifactAside {
    param([Parameter(Mandatory = $true)][int]$Fold)
    $artifact = Get-ArtifactPath $Fold
    $resolvedArtifact = [System.IO.Path]::GetFullPath($artifact)
    $resolvedV22Root = [System.IO.Path]::GetFullPath(
        (Join-Path $projectPath 'artifacts\RADAR-LN3-V22-cq08-known-gate-fixed-parent-balanced-oof')
    ).TrimEnd([System.IO.Path]::DirectorySeparatorChar) +
        [System.IO.Path]::DirectorySeparatorChar
    if (-not $resolvedArtifact.StartsWith(
        $resolvedV22Root,
        [System.StringComparison]::OrdinalIgnoreCase
    )) {
        throw ('Refusing to archive a fold outside the V22 artifact root: ' + $resolvedArtifact)
    }
    if (-not (Test-NonEmptyDirectory $resolvedArtifact)) {
        return $null
    }
    $files = @(
        Get-ChildItem -LiteralPath $resolvedArtifact -Recurse -Force -File
    )
    $checkpointFiles = @($files | Where-Object { $_.Extension -eq '.pt' })
    if ($checkpointFiles.Count -gt 0) {
        throw (
            'V22 fold ' + $Fold + ' has checkpoint payloads but no resumable ' +
            'last_model.pt; manual inspection is required: ' + $resolvedArtifact
        )
    }
    $allowedNames = @('training.log', 'failure.json')
    $unexpectedFiles = @($files | Where-Object { $allowedNames -notcontains $_.Name })
    if ($unexpectedFiles.Count -gt 0) {
        throw (
            'V22 fold ' + $Fold + ' pre-checkpoint artifact contains unexpected ' +
            'files and will not be archived automatically: ' +
            (($unexpectedFiles.FullName) -join ', ')
        )
    }
    $archivePath = $resolvedArtifact + '-interrupted-before-first-checkpoint-' +
        (Get-Date -Format 'yyyyMMdd-HHmmss')
    if (Test-Path -LiteralPath $archivePath) {
        throw ('V22 pre-checkpoint archive destination already exists: ' + $archivePath)
    }
    Move-Item -LiteralPath $resolvedArtifact -Destination $archivePath
    return $archivePath
}

function Assert-AllFoldsRequested {
    $normalized = @($Folds | Sort-Object -Unique)
    if (($normalized -join ',') -ne ($allFolds -join ',')) {
        throw 'This stage requires all folds 0,1,2,3,4.'
    }
}

function Assert-Readiness {
    Assert-File $readinessPath 'V22 OOF readiness'
    $readiness = Read-Utf8Json $readinessPath
    $failedChecks = @(
        $readiness.checks.PSObject.Properties |
            Where-Object { $_.Value -ne $true }
    )
    if (
        $readiness.status -ne 'passed' -or
        $readiness.schema_version -ne $readinessSchema -or
        [int]$readiness.fold_count -ne 5 -or
        [int]$readiness.known_parent_count -ne 986 -or
        [int]$readiness.provenance_component_count -ne 975 -or
        [int]$readiness.selected_seed -ne 43 -or
        $readiness.validation_used_for_readiness_gate -ne $false -or
        [int]$readiness.test_payload_access_count -ne 0 -or
        $readiness.deployment_eligible -ne $false -or
        $failedChecks.Count -ne 0
    ) {
        throw ('V22 OOF readiness is stale or failed: ' + $readinessPath)
    }
}

function Assert-CompletedFold {
    param([Parameter(Mandatory = $true)][int]$Fold)
    $artifact = Get-ArtifactPath $Fold
    $predictionPath = Join-Path $artifact 'held-out-predictions.json'
    $summaryPath = Join-Path $artifact 'training-summary.json'
    $checkpointPath = Join-Path $artifact 'final_model.pt'
    Assert-File $predictionPath ('V22 fold ' + $Fold + ' prediction')
    Assert-File $summaryPath ('V22 fold ' + $Fold + ' summary')
    Assert-File $checkpointPath ('V22 fold ' + $Fold + ' final checkpoint')
    $prediction = Read-Utf8Json $predictionPath
    $summary = Read-Utf8Json $summaryPath
    if (
        $prediction.status -ne 'passed' -or
        $prediction.schema_version -ne $predictionSchema -or
        [int]$prediction.fold_id -ne $Fold -or
        [int]$prediction.epochs -ne 20 -or
        $prediction.loss_aggregation.class_weight_normalization -ne 'fixed_parent_count' -or
        $prediction.held_out_checkpoint_selection -ne $false -or
        [int]$prediction.unknown_training_exposure -ne 0 -or
        [int]$prediction.validation_access_count -ne 0 -or
        [int]$prediction.test_payload_access_count -ne 0 -or
        @($prediction.records.parent).Count -ne [int]$prediction.held_out_scope.parent_count -or
        $summary.status -ne 'completed' -or
        [int]$summary.epoch -ne 20 -or
        $summary.deployment_eligible -ne $false
    ) {
        throw ('V22 fold ' + $Fold + ' is incomplete or outside the contract')
    }
}

function Invoke-Prepare {
    Write-Host '[V22] preparing fixed-parent-denominator loss-only OOF' -ForegroundColor Green
    Invoke-RadarPython @(
        '-m', 'py_compile',
        'radar_ln3\losses.py',
        'radar_ln3\known_gate.py',
        'radar_ln3\known_gate_v22_oof.py',
        'build\known_gate_v22_oof.py',
        'train\known_gate_v22_oof.py',
        'report\known_gate_v22_oof.py'
    )
    Invoke-RadarPython @(
        '-m', 'build.known_gate_v22_oof',
        '--project-root', $projectPath
    )
    Assert-Readiness
    Invoke-RadarPython @(
        '--project-root', $projectPath
    )
    Invoke-RadarPython @(
        '--root', $srcRoot
    )
}

function Invoke-FoldTraining {
    param([Parameter(Mandatory = $true)][int]$Fold)
    Assert-Readiness
    $artifact = Get-ArtifactPath $Fold
    $predictionPath = Join-Path $artifact 'held-out-predictions.json'
    $lastCheckpoint = Join-Path $artifact 'last_model.pt'
    if (Test-Path -LiteralPath $predictionPath -PathType Leaf) {
        Assert-CompletedFold $Fold
        Write-Host ('[V22] fold ' + $Fold + ' already completed.') -ForegroundColor Yellow
        return
    }
    $arguments = @(
        '-m', 'train.known_gate_v22_oof',
        '--project-root', $projectPath,
        '--fold-id', [string]$Fold
    )
    $restartFromFrozenInitialState = $false
    if (Test-NonEmptyDirectory $artifact) {
        if (Test-Path -LiteralPath $lastCheckpoint -PathType Leaf) {
            $arguments += @('--resume-checkpoint', $lastCheckpoint)
            Write-Host ('[V22] resuming fold ' + $Fold) -ForegroundColor Yellow
        }
        elseif (@($RestartPreCheckpointFolds) -contains $Fold) {
            $restartFromFrozenInitialState = $true
        }
        else {
            throw (
                'V22 fold ' + $Fold + ' stopped before its first epoch checkpoint. ' +
                'The artifact remains untouched. After resolving the external cause, ' +
                'rerun with -RestartPreCheckpointFolds ' + $Fold + ' to archive the ' +
                'failed attempt and restart from the frozen initial state.'
            )
        }
    }
    Assert-GpuMemoryHeadroom $Fold
    if ($restartFromFrozenInitialState) {
        $archivePath = Move-PreCheckpointFoldArtifactAside $Fold
        if ([string]::IsNullOrWhiteSpace([string]$archivePath)) {
            throw ('V22 fold ' + $Fold + ' pre-checkpoint artifact was not archived.')
        }
        Write-Host (
            '[V22] archived fold ' + $Fold + ' pre-checkpoint attempt and will ' +
            'restart from the frozen initial state: ' + $archivePath
        ) -ForegroundColor Yellow
    }
    Invoke-RadarPython $arguments
    Assert-CompletedFold $Fold
}

function Invoke-Treatment {
    foreach ($fold in @($Folds | Sort-Object -Unique)) {
        Write-Host ('[V22] full 20-epoch Treatment fold ' + $fold) -ForegroundColor Green
        Invoke-FoldTraining $fold
    }
}

function Invoke-Evaluate {
    Assert-AllFoldsRequested
    foreach ($fold in $allFolds) {
        Assert-CompletedFold $fold
    }
    Invoke-RadarPython @(
        '-m', 'report.known_gate_v22_oof',
        '--project-root', $projectPath
    )
    Assert-File $reportPath 'V22 paired OOF report'
    $report = Read-Utf8Json $reportPath
    if (
        $report.schema_version -ne $reportSchema -or
        $report.deployment_eligible -ne $false -or
        $report.decision.advance_to_l3_calibration -ne $false -or
        $report.decision.online_known_gate_adoption_allowed -ne $false
    ) {
        throw ('V22 paired report violates its contract: ' + $reportPath)
    }
}

try {
    Write-Host ('[V22] project=' + $projectPath) -ForegroundColor Green
    Write-Host ('[V22] stage=' + $Stage + '; folds=' + (($Folds | Sort-Object -Unique) -join ',')) -ForegroundColor Green
    switch ($Stage) {
        'prepare' { Invoke-Prepare }
        'treatment' { Invoke-Treatment }
        'evaluate' { Invoke-Evaluate }
        'all' {
            Assert-AllFoldsRequested
            Invoke-Prepare
            Invoke-Treatment
            Invoke-Evaluate
        }
    }
    Write-Host '[V22] requested stages completed. Unknown, threshold fitting, test, cascade, stable and P95 remain unexecuted.' -ForegroundColor Green
}
catch {
    [Console]::Error.WriteLine('[V22] pipeline stopped: ' + $_.Exception.Message)
    exit 1
}
