[CmdletBinding()]
param(
    [ValidateSet('prepare', 'treatment', 'evaluate', 'all')]
    [string]$Stage = 'all',
    [ValidateSet('v23a', 'v23b', 'both')]
    [string]$Arm = 'both',
    [ValidateRange(0, 4)]
    [int[]]$Folds = @(0, 1, 2, 3, 4),
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
$defaultProjectRoot = Split-Path -Parent $srcRoot
$projectPath = [System.IO.Path]::GetFullPath(
    $(if ([string]::IsNullOrWhiteSpace($ProjectRoot)) { $defaultProjectRoot } else { $ProjectRoot })
)
$python = Join-Path $projectPath '.venv\Scripts\python.exe'
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

function Assert-AllFoldsRequested {
    $normalized = @($Folds | Sort-Object -Unique)
    if (($normalized -join ',') -ne ($allFolds -join ',')) {
        throw 'The evaluate and all stages require folds 0,1,2,3,4.'
    }
}

function Get-ArmsToRun {
    if ($Arm -eq 'both') {
        return @('v23a', 'v23b')
    }
    return @($Arm)
}

function Get-ArtifactPath {
    param(
        [Parameter(Mandatory = $true)][string]$ExperimentArm,
        [Parameter(Mandatory = $true)][int]$Fold
    )
    $slug = if ($ExperimentArm -eq 'v23a') { 'v21e-loss' } else { 'v22-fixed-parent-balanced-loss' }
    return Join-Path $projectPath (
        'artifacts\RADAR-LN3-' + $ExperimentArm.ToUpperInvariant() +
        '-cq08-known-gate-official-cfc-' + $slug + '-oof\fold-' +
        $Fold.ToString('00') + '\treatment'
    )
}

function Get-ReadinessPath {
    param([Parameter(Mandatory = $true)][string]$ExperimentArm)
    $slug = if ($ExperimentArm -eq 'v23a') { 'v21e-loss' } else { 'v22-fixed-parent-balanced-loss' }
    return Join-Path $projectPath (
        'src-v3\processed\report\RADAR-LN3-' + $ExperimentArm.ToUpperInvariant() +
        '-known-gate-official-cfc-' + $slug + '-oof-training-readiness.json'
    )
}

function Get-ReportPath {
    param([Parameter(Mandatory = $true)][string]$ExperimentArm)
    $slug = if ($ExperimentArm -eq 'v23a') { 'v21e-loss' } else { 'v22-fixed-parent-balanced-loss' }
    return Join-Path $projectPath (
        'src-v3\processed\report\RADAR-LN3-' + $ExperimentArm.ToUpperInvariant() +
        '-known-gate-official-cfc-' + $slug + '-paired-oof.json'
    )
}

function Get-GpuMemoryState {
    $nvidiaSmiCommand = Get-Command 'nvidia-smi.exe' -ErrorAction SilentlyContinue
    if ($null -eq $nvidiaSmiCommand) {
        throw 'nvidia-smi.exe is unavailable; V23 cannot audit CUDA memory headroom.'
    }
    $queryOutput = @(
        & $nvidiaSmiCommand.Source @(
            '--id=0',
            '--query-gpu=memory.free,memory.total',
            '--format=csv,noheader,nounits'
        )
    )
    if ([int]$LASTEXITCODE -ne 0 -or $queryOutput.Count -ne 1) {
        throw 'Unable to query GPU 0 memory with nvidia-smi.'
    }
    $parts = @([string]$queryOutput[0] -split ',')
    if ($parts.Count -ne 2) {
        throw ('Unexpected nvidia-smi memory response: ' + [string]$queryOutput[0])
    }
    return [pscustomobject]@{
        FreeMiB = [int]::Parse($parts[0].Trim(), [System.Globalization.CultureInfo]::InvariantCulture)
        TotalMiB = [int]::Parse($parts[1].Trim(), [System.Globalization.CultureInfo]::InvariantCulture)
    }
}

function Assert-GpuMemoryHeadroom {
    param(
        [Parameter(Mandatory = $true)][string]$ExperimentArm,
        [Parameter(Mandatory = $true)][int]$Fold
    )
    $memory = Get-GpuMemoryState
    Write-Host (
        '[V23] arm=' + $ExperimentArm + '; fold=' + $Fold + '; GPU0 free=' +
        $memory.FreeMiB + ' MiB; required=' + $minimumFreeGpuMemoryMiB + ' MiB'
    ) -ForegroundColor Cyan
    if ($memory.FreeMiB -lt $minimumFreeGpuMemoryMiB) {
        throw ('V23 CUDA preflight blocked: free GPU memory is below ' + $minimumFreeGpuMemoryMiB + ' MiB.')
    }
}

function Assert-Readiness {
    param([Parameter(Mandatory = $true)][string]$ExperimentArm)
    $path = Get-ReadinessPath $ExperimentArm
    Assert-File $path ('V23 ' + $ExperimentArm + ' readiness')
    $readiness = Read-Utf8Json $path
    $failedChecks = @(
        $readiness.checks.PSObject.Properties |
            Where-Object { $_.Value -ne $true }
    )
    $expectedSchema = 'RADAR-LN3-' + $ExperimentArm.ToUpperInvariant() + '-known-gate-official-cfc-oof-readiness-v1'
    if (
        $readiness.status -ne 'passed' -or
        $readiness.arm -ne $ExperimentArm -or
        $readiness.schema_version -ne $expectedSchema -or
        [int]$readiness.fold_count -ne 5 -or
        [int]$readiness.known_parent_count -ne 986 -or
        [int]$readiness.provenance_component_count -ne 975 -or
        [int]$readiness.selected_seed -ne 43 -or
        $readiness.validation_used_for_readiness_gate -ne $false -or
        [int]$readiness.test_payload_access_count -ne 0 -or
        $readiness.deployment_eligible -ne $false -or
        $failedChecks.Count -ne 0
    ) {
        throw ('V23 readiness is stale or failed: ' + $path)
    }
}

function Assert-CompletedFold {
    param(
        [Parameter(Mandatory = $true)][string]$ExperimentArm,
        [Parameter(Mandatory = $true)][int]$Fold
    )
    $artifact = Get-ArtifactPath $ExperimentArm $Fold
    $predictionPath = Join-Path $artifact 'held-out-predictions.json'
    $summaryPath = Join-Path $artifact 'training-summary.json'
    $checkpointPath = Join-Path $artifact 'final_model.pt'
    Assert-File $predictionPath ('V23 ' + $ExperimentArm + ' fold ' + $Fold + ' prediction')
    Assert-File $summaryPath ('V23 ' + $ExperimentArm + ' fold ' + $Fold + ' summary')
    Assert-File $checkpointPath ('V23 ' + $ExperimentArm + ' fold ' + $Fold + ' final checkpoint')
    $prediction = Read-Utf8Json $predictionPath
    $summary = Read-Utf8Json $summaryPath
    $schema = 'RADAR-LN3-' + $ExperimentArm.ToUpperInvariant() + '-known-gate-official-cfc-oof-prediction-v1'
    if (
        $prediction.status -ne 'passed' -or
        $prediction.schema_version -ne $schema -or
        $prediction.arm -ne $ExperimentArm -or
        $prediction.architecture -ne 'official_ncps_cfc' -or
        [int]$prediction.fold_id -ne $Fold -or
        [int]$prediction.epochs -ne 20 -or
        $prediction.held_out_checkpoint_selection -ne $false -or
        [int]$prediction.unknown_training_exposure -ne 0 -or
        [int]$prediction.validation_access_count -ne 0 -or
        [int]$prediction.test_payload_access_count -ne 0 -or
        $prediction.official_cfc.distribution -ne 'ncps' -or
        $prediction.official_cfc.distribution_version -ne '1.0.1' -or
        $prediction.official_cfc.git_commit -ne '695ec22d7d1831f2ced67346ef7d2f08c525bd47' -or
        $summary.status -ne 'completed' -or
        [int]$summary.epoch -ne 20 -or
        $summary.deployment_eligible -ne $false
    ) {
        throw ('V23 ' + $ExperimentArm + ' fold ' + $Fold + ' is incomplete or outside its contract.')
    }
}

function Invoke-Prepare {
    param([Parameter(Mandatory = $true)][string]$ExperimentArm)
    Write-Host ('[V23] preparing official CfC arm=' + $ExperimentArm) -ForegroundColor Green
    Invoke-RadarPython @(
        '-m', 'py_compile',
        'radar_ln3\official_cfc.py',
        'radar_ln3\known_gate_official_cfc.py',
        'radar_ln3\known_gate_v23.py',
        'radar_ln3\known_gate_v23_oof.py',
        'build\known_gate_v23_oof.py',
        'train\known_gate_v23_oof.py',
        'report\known_gate_v23_oof.py',
        'smoke\known_gate_v23_official_cfc_contract_test.py'
    )
    Invoke-RadarPython @(
        '-m', 'build.known_gate_v23_oof',
        '--project-root', $projectPath,
        '--arm', $ExperimentArm
    )
    Assert-Readiness $ExperimentArm
    Invoke-RadarPython @(
        '-m', 'smoke.known_gate_v23_official_cfc_contract_test',
        '--project-root', $projectPath,
        '--arm', $ExperimentArm
    )
    Invoke-RadarPython @(
        '-m', 'smoke.static_isolation_check',
        '--root', $srcRoot
    )
}

function Invoke-FoldTraining {
    param(
        [Parameter(Mandatory = $true)][string]$ExperimentArm,
        [Parameter(Mandatory = $true)][int]$Fold
    )
    Assert-Readiness $ExperimentArm
    $artifact = Get-ArtifactPath $ExperimentArm $Fold
    $predictionPath = Join-Path $artifact 'held-out-predictions.json'
    $lastCheckpoint = Join-Path $artifact 'last_model.pt'
    if (Test-Path -LiteralPath $predictionPath -PathType Leaf) {
        Assert-CompletedFold $ExperimentArm $Fold
        Write-Host ('[V23] arm=' + $ExperimentArm + '; fold=' + $Fold + ' already completed.') -ForegroundColor Yellow
        return
    }
    $arguments = @(
        '-m', 'train.known_gate_v23_oof',
        '--project-root', $projectPath,
        '--arm', $ExperimentArm,
        '--fold-id', [string]$Fold
    )
    if (Test-Path -LiteralPath $artifact -PathType Container) {
        if (Test-Path -LiteralPath $lastCheckpoint -PathType Leaf) {
            $arguments += @('--resume-checkpoint', $lastCheckpoint)
            Write-Host ('[V23] resuming arm=' + $ExperimentArm + '; fold=' + $Fold) -ForegroundColor Yellow
        }
        elseif ($null -ne (Get-ChildItem -LiteralPath $artifact -Force | Select-Object -First 1)) {
            throw (
                'V23 arm=' + $ExperimentArm + '; fold=' + $Fold +
                ' contains a non-resumable pre-checkpoint attempt. Preserve it for inspection, then manually archive it before retrying.'
            )
        }
    }
    Assert-GpuMemoryHeadroom $ExperimentArm $Fold
    Invoke-RadarPython $arguments
    Assert-CompletedFold $ExperimentArm $Fold
}

function Invoke-Treatment {
    param([Parameter(Mandatory = $true)][string]$ExperimentArm)
    foreach ($fold in @($Folds | Sort-Object -Unique)) {
        Write-Host ('[V23] full 20-epoch official CfC arm=' + $ExperimentArm + '; fold=' + $fold) -ForegroundColor Green
        Invoke-FoldTraining $ExperimentArm $fold
    }
}

function Invoke-Evaluate {
    param([Parameter(Mandatory = $true)][string]$ExperimentArm)
    Assert-AllFoldsRequested
    foreach ($fold in $allFolds) {
        Assert-CompletedFold $ExperimentArm $fold
    }
    Invoke-RadarPython @(
        '-m', 'report.known_gate_v23_oof',
        '--project-root', $projectPath,
        '--arm', $ExperimentArm
    )
    $path = Get-ReportPath $ExperimentArm
    Assert-File $path ('V23 ' + $ExperimentArm + ' paired OOF report')
    $report = Read-Utf8Json $path
    $schema = 'RADAR-LN3-' + $ExperimentArm.ToUpperInvariant() + '-known-gate-official-cfc-paired-oof-report-v1'
    if (
        $report.schema_version -ne $schema -or
        $report.deployment_eligible -ne $false -or
        $report.decision.advance_to_l3_calibration -ne $false -or
        $report.decision.online_known_gate_adoption_allowed -ne $false
    ) {
        throw ('V23 ' + $ExperimentArm + ' paired report violates its contract: ' + $path)
    }
}

try {
    Write-Host ('[V23] project=' + $projectPath) -ForegroundColor Green
    Write-Host ('[V23] stage=' + $Stage + '; arm=' + $Arm + '; folds=' + (($Folds | Sort-Object -Unique) -join ',')) -ForegroundColor Green
    foreach ($experimentArm in Get-ArmsToRun) {
        switch ($Stage) {
            'prepare' { Invoke-Prepare $experimentArm }
            'treatment' { Invoke-Treatment $experimentArm }
            'evaluate' { Invoke-Evaluate $experimentArm }
            'all' {
                Assert-AllFoldsRequested
                Invoke-Prepare $experimentArm
                Invoke-Treatment $experimentArm
                Invoke-Evaluate $experimentArm
            }
        }
    }
    Write-Host '[V23] requested stages completed. Unknown, threshold fitting, test, cascade, stable and P95 remain unexecuted.' -ForegroundColor Green
}
catch {
    [Console]::Error.WriteLine('[V23] pipeline stopped: ' + $_.Exception.Message)
    exit 1
}
