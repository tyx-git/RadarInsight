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
$projectPath = [System.IO.Path]::GetFullPath($(if ([string]::IsNullOrWhiteSpace($ProjectRoot)) { $defaultProjectRoot } else { $ProjectRoot }))
$python = Join-Path $projectPath '.venv\Scripts\python.exe'
$readinessPath = Join-Path $projectPath 'src-v3\processed\report\RADAR-LN3-V22b-known-gate-rd-temporal-late-readout-oof-training-readiness.json'
$reportPath = Join-Path $projectPath 'src-v3\processed\report\RADAR-LN3-V22b-known-gate-rd-temporal-late-readout-paired-oof.json'
$readinessSchema = 'RADAR-LN3-V22b-known-gate-rd-temporal-late-readout-oof-readiness-v1'
$predictionSchema = 'RADAR-LN3-V22b-known-gate-rd-temporal-late-readout-oof-prediction-v1'
$reportSchema = 'RADAR-LN3-V22b-known-gate-rd-temporal-late-readout-paired-oof-report-v1'
$allFolds = @(0, 1, 2, 3, 4)
$minimumFreeGpuMemoryMiB = 4096

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) { throw ('Project Python interpreter is missing: ' + $python) }

function Invoke-RadarPython {
    param([Parameter(Mandatory = $true)][string[]]$Arguments)
    $runtimeArguments = @('-X', 'utf8') + $Arguments
    $display = $python + ' ' + ($runtimeArguments -join ' ')
    Write-Host ('> ' + $display) -ForegroundColor Cyan
    Push-Location $srcRoot
    try { & $python @runtimeArguments | Out-Host; $exitCode = [int]$LASTEXITCODE }
    finally { Pop-Location }
    if ($exitCode -ne 0) { throw ('Python command failed with exit code ' + $exitCode + ': ' + $display) }
}

function Read-Utf8Json {
    param([Parameter(Mandatory = $true)][string]$Path)
    return ConvertFrom-Json -InputObject ([System.IO.File]::ReadAllText($Path, [System.Text.UTF8Encoding]::new($false, $true)))
}

function Assert-File {
    param([Parameter(Mandatory = $true)][string]$Path, [Parameter(Mandatory = $true)][string]$Description)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { throw ($Description + ' is missing: ' + $Path) }
}

function Get-ArtifactPath {
    param([Parameter(Mandatory = $true)][int]$Fold)
    return Join-Path $projectPath ('artifacts\RADAR-LN3-V22b-cq08-known-gate-rd-temporal-late-readout-paired-oof\fold-' + $Fold.ToString('00') + '\treatment')
}

function Test-NonEmptyDirectory {
    param([Parameter(Mandatory = $true)][string]$Path)
    return (Test-Path -LiteralPath $Path -PathType Container) -and
        ($null -ne (Get-ChildItem -LiteralPath $Path -Force | Select-Object -First 1))
}

function Move-PreCheckpointArtifactAside {
    param([Parameter(Mandatory = $true)][int]$Fold)
    $artifact = [System.IO.Path]::GetFullPath((Get-ArtifactPath $Fold))
    $rootPath = [System.IO.Path]::GetFullPath((Join-Path $projectPath 'artifacts\RADAR-LN3-V22b-cq08-known-gate-rd-temporal-late-readout-paired-oof')).TrimEnd([System.IO.Path]::DirectorySeparatorChar) + [System.IO.Path]::DirectorySeparatorChar
    if (-not $artifact.StartsWith($rootPath, [System.StringComparison]::OrdinalIgnoreCase)) { throw ('Refusing to archive outside V22b artifact root: ' + $artifact) }
    $files = @(Get-ChildItem -LiteralPath $artifact -Recurse -Force -File)
    if (@($files | Where-Object { $_.Extension -eq '.pt' }).Count -gt 0) { throw ('V22b fold ' + $Fold + ' contains checkpoint files but lacks last_model.pt; inspect manually.') }
    $unexpected = @($files | Where-Object { @('training.log','failure.json') -notcontains $_.Name })
    if ($unexpected.Count -gt 0) { throw ('V22b pre-checkpoint artifact contains unexpected files: ' + (($unexpected.FullName) -join ', ')) }
    $destination = $artifact + '-interrupted-before-first-checkpoint-' + (Get-Date -Format 'yyyyMMdd-HHmmss')
    if (Test-Path -LiteralPath $destination) { throw ('V22b archive destination exists: ' + $destination) }
    Move-Item -LiteralPath $artifact -Destination $destination
    return $destination
}

function Assert-AllFoldsRequested {
    if ((@($Folds | Sort-Object -Unique) -join ',') -ne ($allFolds -join ',')) { throw 'Evaluation/all requires folds 0,1,2,3,4.' }
}

function Assert-Readiness {
    Assert-File $readinessPath 'V22b OOF readiness'
    $readiness = Read-Utf8Json $readinessPath
    $failed = @($readiness.checks.PSObject.Properties | Where-Object { $_.Value -ne $true })
    if ($readiness.status -ne 'passed' -or $readiness.schema_version -ne $readinessSchema -or [int]$readiness.fold_count -ne 5 -or [int]$readiness.known_parent_count -ne 986 -or [int]$readiness.provenance_component_count -ne 975 -or [int]$readiness.selected_seed -ne 43 -or $readiness.validation_used_for_readiness_gate -ne $false -or [int]$readiness.unknown_training_exposure -ne 0 -or [int]$readiness.test_payload_access_count -ne 0 -or $readiness.deployment_eligible -ne $false -or $failed.Count -ne 0) {
        throw ('V22b OOF readiness is stale or failed: ' + $readinessPath)
    }
}

function Assert-GpuMemory {
    param([Parameter(Mandatory = $true)][int]$Fold)
    $command = Get-Command 'nvidia-smi.exe' -ErrorAction SilentlyContinue
    if ($null -eq $command) { throw 'nvidia-smi.exe is unavailable.' }
    $row = @(& $command.Source '--id=0' '--query-gpu=memory.free' '--format=csv,noheader,nounits')
    if ([int]$LASTEXITCODE -ne 0 -or $row.Count -ne 1) { throw 'Unable to query GPU memory.' }
    $freeMiB = [int]::Parse(([string]$row[0]).Trim(), [System.Globalization.CultureInfo]::InvariantCulture)
    Write-Host ('[V22b] fold ' + $Fold + ' GPU free=' + $freeMiB + ' MiB') -ForegroundColor Cyan
    if ($freeMiB -lt $minimumFreeGpuMemoryMiB) { throw ('V22b fold ' + $Fold + ' requires at least ' + $minimumFreeGpuMemoryMiB + ' MiB free GPU memory.') }
}

function Assert-CompletedFold {
    param([Parameter(Mandatory = $true)][int]$Fold)
    $artifact = Get-ArtifactPath $Fold
    $predictionPath = Join-Path $artifact 'held-out-predictions.json'
    $summaryPath = Join-Path $artifact 'training-summary.json'
    $checkpointPath = Join-Path $artifact 'final_model.pt'
    Assert-File $predictionPath ('V22b fold ' + $Fold + ' prediction')
    Assert-File $summaryPath ('V22b fold ' + $Fold + ' summary')
    Assert-File $checkpointPath ('V22b fold ' + $Fold + ' checkpoint')
    $prediction = Read-Utf8Json $predictionPath
    $summary = Read-Utf8Json $summaryPath
    if ($prediction.status -ne 'passed' -or $prediction.schema_version -ne $predictionSchema -or [int]$prediction.fold_id -ne $Fold -or [int]$prediction.epochs -ne 20 -or $prediction.held_out_checkpoint_selection -ne $false -or [int]$prediction.unknown_training_exposure -ne 0 -or [int]$prediction.validation_access_count -ne 0 -or [int]$prediction.test_payload_access_count -ne 0 -or $prediction.paired_initialization.initial_logits_exactly_equal -ne $true -or $prediction.paired_initialization.all_model_parameters_trainable -ne $true -or $prediction.paired_initialization.liquid_state_receives_residual -ne $false -or $summary.status -ne 'completed' -or [int]$summary.epoch -ne 20 -or $summary.deployment_eligible -ne $false) {
        throw ('V22b fold ' + $Fold + ' is incomplete or violates contract')
    }
}

function Invoke-Prepare {
    Write-Host '[V22b] preparing RD temporal late-readout paired OOF' -ForegroundColor Green
    Invoke-RadarPython @('-m','py_compile','radar_ln3\known_gate.py','radar_ln3\known_gate_v22b.py','radar_ln3\known_gate_v22b_oof.py','build\known_gate_v22b_oof.py','train\known_gate_v22b_oof.py','report\known_gate_v22b_oof.py')
    Invoke-RadarPython @('-m','build.known_gate_v22b_oof','--project-root',$projectPath)
    Assert-Readiness
}

function Invoke-Fold {
    param([Parameter(Mandatory = $true)][int]$Fold)
    Assert-Readiness
    $artifact = Get-ArtifactPath $Fold
    $prediction = Join-Path $artifact 'held-out-predictions.json'
    $last = Join-Path $artifact 'last_model.pt'
    if (Test-Path -LiteralPath $prediction -PathType Leaf) { Assert-CompletedFold $Fold; Write-Host ('[V22b] fold ' + $Fold + ' already completed.') -ForegroundColor Yellow; return }
    $arguments = @('-m','train.known_gate_v22b_oof','--project-root',$projectPath,'--fold-id',[string]$Fold)
    $restart = $false
    if (Test-NonEmptyDirectory $artifact) {
        if (Test-Path -LiteralPath $last -PathType Leaf) {
            $arguments += @('--resume-checkpoint',$last)
            Write-Host ('[V22b] resuming fold ' + $Fold) -ForegroundColor Yellow
        }
        elseif (@($RestartPreCheckpointFolds) -contains $Fold) {
            $restart = $true
        }
        else {
            throw ('V22b fold ' + $Fold + ' stopped before its first checkpoint. Rerun with -RestartPreCheckpointFolds ' + $Fold + ' after resolving the external cause.')
        }
    }
    Assert-GpuMemory $Fold
    if ($restart) {
        $archive = Move-PreCheckpointArtifactAside $Fold
        Write-Host ('[V22b] archived pre-checkpoint attempt: ' + $archive) -ForegroundColor Yellow
    }
    Invoke-RadarPython $arguments
    Assert-CompletedFold $Fold
}

function Invoke-Treatment {
    foreach ($fold in @($Folds | Sort-Object -Unique)) { Write-Host ('[V22b] full 20-epoch Treatment fold ' + $fold) -ForegroundColor Green; Invoke-Fold $fold }
}

function Invoke-Evaluate {
    Assert-AllFoldsRequested
    foreach ($fold in $allFolds) { Assert-CompletedFold $fold }
    Invoke-RadarPython @('-m','report.known_gate_v22b_oof','--project-root',$projectPath)
    Assert-File $reportPath 'V22b paired OOF report'
    $report = Read-Utf8Json $reportPath
    if ($report.schema_version -ne $reportSchema -or $report.deployment_eligible -ne $false -or $report.decision.advance_to_l3_calibration -ne $false -or $report.decision.online_known_gate_adoption_allowed -ne $false) { throw ('V22b report violates contract: ' + $reportPath) }
}

try {
    Write-Host ('[V22b] project=' + $projectPath) -ForegroundColor Green
    Write-Host ('[V22b] stage=' + $Stage + '; folds=' + (($Folds | Sort-Object -Unique) -join ',')) -ForegroundColor Green
    switch ($Stage) {
        'prepare' { Invoke-Prepare }
        'treatment' { Invoke-Treatment }
        'evaluate' { Invoke-Evaluate }
        'all' { Assert-AllFoldsRequested; Invoke-Prepare; Invoke-Treatment; Invoke-Evaluate }
    }
    Write-Host '[V22b] requested stages completed. Unknown, threshold fitting, test, cascade, stable and P95 remain unexecuted.' -ForegroundColor Green
}
catch { [Console]::Error.WriteLine('[V22b] pipeline stopped: ' + $_.Exception.Message); exit 1 }
