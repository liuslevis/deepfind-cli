param(
    [switch]$SeparateWindows,
    [switch]$SkipSetup,
    [switch]$SkipXhsLogin
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

function Test-NeedsPythonSync {
    $venvPath = Join-Path $repoRoot '.venv'
    $syncMarker = Join-Path $venvPath '.uv-sync-marker'
    $pyprojectPath = Join-Path $repoRoot 'pyproject.toml'
    $uvLockPath = Join-Path $repoRoot 'uv.lock'

    # If venv doesn't exist, we need to sync
    if (-not (Test-Path -LiteralPath $venvPath)) {
        return $true
    }

    # If marker doesn't exist, we need to sync
    if (-not (Test-Path -LiteralPath $syncMarker)) {
        return $true
    }

    $markerTime = (Get-Item -LiteralPath $syncMarker).LastWriteTime

    # Check if pyproject.toml is newer than marker
    if ((Test-Path -LiteralPath $pyprojectPath) -and ((Get-Item -LiteralPath $pyprojectPath).LastWriteTime -gt $markerTime)) {
        return $true
    }

    # Check if uv.lock is newer than marker
    if ((Test-Path -LiteralPath $uvLockPath) -and ((Get-Item -LiteralPath $uvLockPath).LastWriteTime -gt $markerTime)) {
        return $true
    }

    return $false
}

function Test-NeedsWebSync {
    $packageJsonPath = Join-Path $webRoot 'package.json'
    $syncMarker = Join-Path $webNodeModules '.npm-sync-marker'

    # If node_modules doesn't exist, we need to sync
    if (-not (Test-Path -LiteralPath $webNodeModules)) {
        return $true
    }

    # If marker doesn't exist, we need to sync
    if (-not (Test-Path -LiteralPath $syncMarker)) {
        return $true
    }

    $markerTime = (Get-Item -LiteralPath $syncMarker).LastWriteTime

    # Check if package.json is newer than marker
    if ((Test-Path -LiteralPath $packageJsonPath) -and ((Get-Item -LiteralPath $packageJsonPath).LastWriteTime -gt $markerTime)) {
        return $true
    }

    # Check if package-lock.json is newer than marker
    if ((Test-Path -LiteralPath $webPackageLock) -and ((Get-Item -LiteralPath $webPackageLock).LastWriteTime -gt $markerTime)) {
        return $true
    }

    return $false
}

function Invoke-SetupStep {
    if ($SkipSetup) {
        return
    }

    Write-Host 'Updating uv tools...' -ForegroundColor Cyan
    & uv tool install --upgrade bilibili-cli
    & uv tool install --upgrade xiaohongshu-cli

    $needsPythonSync = Test-NeedsPythonSync
    if ($needsPythonSync) {
        Write-Host 'Syncing Python dependencies with uv...' -ForegroundColor Cyan
        Push-Location $repoRoot
        try {
            & uv sync --extra media --extra local-llm --extra browser

            # Create sync marker
            $venvPath = Join-Path $repoRoot '.venv'
            $syncMarker = Join-Path $venvPath '.uv-sync-marker'
            New-Item -Path $syncMarker -ItemType File -Force | Out-Null
        }
        finally {
            Pop-Location
        }
    }
    else {
        Write-Host 'Python dependencies are up to date, skipping sync.' -ForegroundColor Green
    }

    # Install Playwright browsers if playwright is available
    Push-Location $repoRoot
    try {
        $playwrightCheck = & uv run python -c "import playwright" 2>$null
        if ($LASTEXITCODE -eq 0) {
            $playwrightCache = Join-Path $env:USERPROFILE 'AppData\Local\ms-playwright'
            if (-not (Test-Path -Path "$playwrightCache\chromium-*")) {
                Write-Host 'Installing Playwright browsers...' -ForegroundColor Cyan
                & uv run playwright install
            }
            else {
                Write-Host 'Playwright browsers already installed.' -ForegroundColor Green
            }
        }
    }
    finally {
        Pop-Location
    }

    $needsWebSync = Test-NeedsWebSync
    if ($needsWebSync) {
        Write-Host 'Installing web dependencies...' -ForegroundColor Cyan
        Push-Location $webRoot
        try {
            if ((Test-Path -LiteralPath $webPackageLock) -and -not (Test-Path -LiteralPath $webNodeModules)) {
                & npm ci
            }
            else {
                & npm install
            }

            # Create sync marker
            $syncMarker = Join-Path $webNodeModules '.npm-sync-marker'
            New-Item -Path $syncMarker -ItemType File -Force | Out-Null
        }
        finally {
            Pop-Location
        }
    }
    else {
        Write-Host 'Web dependencies are up to date, skipping install.' -ForegroundColor Green
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

if (-not $SkipXhsLogin -and -not (Test-CommandAvailable -Name 'xhs')) {
    throw 'xhs is not available in PATH. Install xiaohongshu-cli or add xhs to PATH first.'
}

if (-not $SkipXhsLogin) {
    Write-Host 'Logging into Xiaohongshu via QR code...' -ForegroundColor Cyan
    & xhs login --qrcode
} else {
    Write-Host 'Skipping Xiaohongshu login.' -ForegroundColor Yellow
}

# Kill any existing processes on the ports
Write-Host 'Checking for existing processes...' -ForegroundColor Cyan

# Kill processes on port 8000 (backend)
$backendProcess = Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique
if ($backendProcess) {
    Write-Host "Killing existing backend process on port 8000 (PID: $backendProcess)..." -ForegroundColor Yellow
    Stop-Process -Id $backendProcess -Force -ErrorAction SilentlyContinue
    Write-Host 'Previous backend process killed.' -ForegroundColor Green
}

# Kill processes on port 5173 (frontend - Vite default)
$frontendProcess = Get-NetTCPConnection -LocalPort 5173 -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique
if ($frontendProcess) {
    Write-Host "Killing existing frontend process on port 5173 (PID: $frontendProcess)..." -ForegroundColor Yellow
    Stop-Process -Id $frontendProcess -Force -ErrorAction SilentlyContinue
    Write-Host 'Previous frontend process killed.' -ForegroundColor Green
}

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
