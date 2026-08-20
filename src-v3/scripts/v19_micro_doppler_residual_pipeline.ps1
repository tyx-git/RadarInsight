param(
    [ValidateSet('prepare', 'train', 'evaluate', 'all')]
    [string]$Stage = 'all',
    [string]$ProjectRoot = 'F:\Project\RadarInsight'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$projectPath = [System.IO.Path]::GetFullPath($ProjectRoot)
$srcPath = Join-Path $projectPath 'src-v3'
$pythonPath = Join-Path $projectPath '.venv\Scripts\python.exe'
$configPath = Join-Path $projectPath 'config\RADAR-LN3\RADAR-LN3-V19-cq08-dbb-specialist-micro-doppler-late-residual-seed42-config.json'
$readinessPath = Join-Path $srcPath 'processed\report\RADAR-LN3-V19-micro-doppler-late-residual-train-readiness.json'
$artifactPath = Join-Path $projectPath 'artifacts\RADAR-LN3-V19-cq08-dbb-specialist-micro-doppler-late-residual-seed42-formal'
$summaryPath = Join-Path $artifactPath 'training-summary.json'
$lastCheckpointPath = Join-Path $artifactPath 'last_model.pt'

if (-not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) {
    throw ('Python executable is missing: ' + $pythonPath)
}
if (-not (Test-Path -LiteralPath $srcPath -PathType Container)) {
    throw ('src-v3 directory is missing: ' + $srcPath)
}

function Invoke-RadarPython {
    param([string[]]$Arguments)

    Write-Host ('> ' + $pythonPath + ' ' + ($Arguments -join ' '))
    Push-Location $srcPath
    try {
        & $pythonPath @Arguments
        if ($LASTEXITCODE -ne 0) {
            throw ('Python command failed with exit code ' + $LASTEXITCODE)
        }
    }
    finally {
        Pop-Location
    }
}

function Test-ArtifactNonEmpty {
    if (-not (Test-Path -LiteralPath $artifactPath -PathType Container)) {
        return $false
    }
    $first = Get-ChildItem -LiteralPath $artifactPath -Force | Select-Object -First 1
    return $null -ne $first
}

function Read-Utf8Json {
    param([Parameter(Mandatory = $true)][string]$Path)

    $text = [System.IO.File]::ReadAllText(
        $Path,
        [System.Text.UTF8Encoding]::new($false, $true)
    )
    return ConvertFrom-Json -InputObject $text
}

function Assert-Prepared {
    if (-not (Test-Path -LiteralPath $configPath -PathType Leaf)) {
        throw ('V19 config is missing: ' + $configPath)
    }
    if (-not (Test-Path -LiteralPath $readinessPath -PathType Leaf)) {
        throw ('V19 readiness is missing: ' + $readinessPath)
    }
    $readiness = Read-Utf8Json -Path $readinessPath
    if ($readiness.status -ne 'passed') {
        throw ('V19 readiness did not pass: ' + $readiness.status)
    }
    if ($null -eq $readiness.test) {
        throw 'V19 readiness does not declare sealed-test status.'
    }
    if ($readiness.deployment_eligible -ne $false) {
        throw 'V19 readiness must not be deployment eligible.'
    }
}

function Invoke-Prepare {
    Write-Host '[V19] stage=prepare'
    Invoke-RadarPython -Arguments @(
        '-m', 'compileall', '-q',
        'radar_ln3\micro_doppler_residual.py',
        'build\micro_doppler_residual_v19.py',
        'train\micro_doppler_residual_v19.py',
        'report\micro_doppler_residual_v19.py',
        'smoke\micro_doppler_residual_v19_contract_test.py'
    )
    Invoke-RadarPython -Arguments @(
        '-m', 'smoke.static_isolation_check',
        '--root', $srcPath
    )
    Invoke-RadarPython -Arguments @(
        '-m', 'smoke.micro_doppler_residual_v19_contract_test',
        '--project-root', $projectPath
    )

    if (Test-ArtifactNonEmpty) {
        Write-Host '[V19] formal artifact is non-empty; immutable prepare will not be rebuilt.'
        Assert-Prepared
    }
    else {
        Invoke-RadarPython -Arguments @(
            '-m', 'build.micro_doppler_residual_v19',
            '--project-root', $projectPath
        )
        Assert-Prepared
    }

    Invoke-RadarPython -Arguments @(
        '-m', 'smoke.micro_doppler_residual_v19_contract_test',
        '--project-root', $projectPath,
        '--prepared'
    )
}

function Invoke-Train {
    Write-Host '[V19] stage=train'
    Assert-Prepared
    if (Test-Path -LiteralPath $summaryPath -PathType Leaf) {
        $summary = Read-Utf8Json -Path $summaryPath
        if ($summary.status -eq 'completed') {
            Write-Host ('[V19] training already completed; best_epoch=' + $summary.best_epoch)
            return
        }
        if (($summary.status -eq 'failed' -or $summary.status -eq 'running') -and (Test-Path -LiteralPath $lastCheckpointPath -PathType Leaf)) {
            Write-Host '[V19] resuming from the last completed epoch checkpoint.'
            Invoke-RadarPython -Arguments @(
                '-m', 'train.micro_doppler_residual_v19',
                '--project-root', $projectPath,
                '--config', $configPath,
                '--resume-checkpoint', $lastCheckpointPath
            )
            return
        }
        throw ('V19 artifact has a non-resumable summary status: ' + $summary.status)
    }
    if (Test-ArtifactNonEmpty) {
        throw ('V19 formal artifact is non-empty without a resumable summary: ' + $artifactPath)
    }
    Invoke-RadarPython -Arguments @(
        '-m', 'train.micro_doppler_residual_v19',
        '--project-root', $projectPath,
        '--config', $configPath
    )
}

function Invoke-Evaluate {
    Write-Host '[V19] stage=evaluate'
    Assert-Prepared
    if (-not (Test-Path -LiteralPath $summaryPath -PathType Leaf)) {
        throw ('V19 training summary is missing: ' + $summaryPath)
    }
    $summary = Read-Utf8Json -Path $summaryPath
    if ($summary.status -ne 'completed') {
        throw ('V19 training is not completed: ' + $summary.status)
    }
    Invoke-RadarPython -Arguments @(
        '-m', 'report.micro_doppler_residual_v19',
        '--project-root', $projectPath,
        '--config', $configPath
    )
}

try {
    Write-Host ('[V19] project=' + $projectPath)
    Write-Host ('[V19] stage=' + $Stage)
    switch ($Stage) {
        'prepare' {
            Invoke-Prepare
        }
        'train' {
            Invoke-Train
        }
        'evaluate' {
            Invoke-Evaluate
        }
        'all' {
            Invoke-Prepare
            Invoke-Train
            Invoke-Evaluate
        }
    }
    Write-Host '[V19] pipeline completed.'
}
catch {
    Write-Error ('V19 micro-Doppler late-residual pipeline stopped: ' + $_.Exception.Message)
    exit 1
}
