param(
    [switch]$SeparateWindows,
    [switch]$SkipSetup
)

$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $PSScriptRoot
$webRoot = Join-Path $repoRoot 'web'
$webNodeModules = Join-Path $webRoot 'node_modules'
$webPackageLock = Join-Path $webRoot 'package-lock.json'

function Test-CommandAvailable {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Name
    )

    return [bool](Get-Command $Name -ErrorAction SilentlyContinue)
}

function Invoke-SetupStep {
    if ($SkipSetup) {
        return
    }

    Write-Host 'Syncing Python dependencies with uv...' -ForegroundColor Cyan
    Push-Location $repoRoot
    try {
        & uv sync --extra media --extra local-llm
    }
    finally {
        Pop-Location
    }

    Write-Host 'Installing web dependencies...' -ForegroundColor Cyan
    Push-Location $webRoot
    try {
        if ((Test-Path -LiteralPath $webPackageLock) -and -not (Test-Path -LiteralPath $webNodeModules)) {
            & npm ci
        }
        else {
            & npm install
        }
    }
    finally {
        Pop-Location
    }
}

if (-not (Test-Path -LiteralPath $webRoot)) {
    throw "Web app directory not found: $webRoot"
}

if (-not (Test-CommandAvailable -Name 'uv')) {
    throw 'uv is not available in PATH. Install uv or add it to PATH first.'
}

if (-not (Test-CommandAvailable -Name 'npm')) {
    throw 'npm is not available in PATH. Install Node.js or add npm to PATH first.'
}

Invoke-SetupStep

$backendCommand = "Set-Location -LiteralPath '$repoRoot'; uv run deepfind-web --reload"
$frontendCommand = "Set-Location -LiteralPath '$webRoot'; npm run dev -- --host"

if (-not $SeparateWindows -and (Test-CommandAvailable -Name 'wt')) {
    Start-Process -FilePath 'wt.exe' -ArgumentList @(
        'new-tab',
        '-d', $repoRoot,
        'powershell.exe',
        '-NoExit',
        '-Command', $backendCommand,
        ';',
        'split-pane',
        '-V',
        '-d', $webRoot,
        'powershell.exe',
        '-NoExit',
        '-Command', $frontendCommand
    )
    return
}

Start-Process -FilePath 'powershell.exe' -WorkingDirectory $repoRoot -ArgumentList @(
    '-NoExit',
    '-Command', $backendCommand
)

Start-Process -FilePath 'powershell.exe' -WorkingDirectory $webRoot -ArgumentList @(
    '-NoExit',
    '-Command', $frontendCommand
)