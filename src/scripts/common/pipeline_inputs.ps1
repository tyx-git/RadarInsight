function Get-PipelineInputHashes {
    param(
        [Parameter(Mandatory = $true)][string]$ProjectRoot,
        [string[]]$Inputs = @()
    )

    $hashes = [ordered]@{}
    foreach ($relative in $Inputs) {
        $path = Join-Path $ProjectRoot $relative
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
            throw "缺少续跑输入文件：$relative"
        }
        $hashes[$relative] = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
    }
    return $hashes
}

function Test-PipelineInputHashes {
    param(
        [object]$SavedHashes,
        [System.Collections.IDictionary]$CurrentHashes
    )

    if ($CurrentHashes.Count -eq 0) { return $true }
    if ($null -eq $SavedHashes) { return $false }
    $savedProperties = @($SavedHashes.PSObject.Properties)
    if ($savedProperties.Count -ne $CurrentHashes.Count) { return $false }
    foreach ($relative in $CurrentHashes.Keys) {
        $savedHash = $savedProperties | Where-Object { $_.Name -eq $relative } | Select-Object -First 1
        if ($null -eq $savedHash -or [string]$savedHash.Value -ne [string]$CurrentHashes[$relative]) {
            return $false
        }
    }
    return $true
}

function Test-PipelineOutputs {
    param(
        [Parameter(Mandatory = $true)][string]$ProjectRoot,
        [Parameter(Mandatory = $true)][string[]]$Outputs,
        [Parameter(Mandatory = $true)][DateTime]$RunStartedUtc,
        [switch]$AllowExisting
    )

    foreach ($relative in $Outputs) {
        $path = Join-Path $ProjectRoot $relative
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { return $false }
        if (-not $AllowExisting -and (Get-Item -LiteralPath $path).LastWriteTimeUtc -lt $RunStartedUtc) {
            return $false
        }
    }
    return $true
}
