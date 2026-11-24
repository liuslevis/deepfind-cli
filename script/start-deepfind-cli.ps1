param(
    [switch]$Restart,
    [switch]$InstallStartup,
    [switch]$RemoveStartup
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$python = Join-Path $repoRoot ".venv\Scripts\python.exe"
$hostAddress = if ($env:DEEPFIND_HOST) { $env:DEEPFIND_HOST } else { "0.0.0.0" }
$port = if ($env:DEEPFIND_PORT) { $env:DEEPFIND_PORT } else { "62000" }
$healthTimeoutSec = if ($env:DEEPFIND_HEALTH_TIMEOUT_SEC) { [int]$env:DEEPFIND_HEALTH_TIMEOUT_SEC } else { 10 }
$startupTimeoutSec = if ($env:DEEPFIND_STARTUP_TIMEOUT_SEC) { [int]$env:DEEPFIND_STARTUP_TIMEOUT_SEC } else { 120 }
$stateDir = Join-Path $env:LOCALAPPDATA "DeepFind"
$logDir = Join-Path $stateDir "logs"
$pidFile = Join-Path $stateDir "deepfind-web.pid"
$stdoutLog = Join-Path $logDir "backend.log"
$stderrLog = Join-Path $logDir "backend-error.log"
$taskName = "DeepFind Web"
$powershell = (Get-Command powershell.exe).Source

if ($InstallStartup -and $RemoveStartup) {
    throw "Use either -InstallStartup or -RemoveStartup, not both."
}

if ($RemoveStartup) {
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction SilentlyContinue
    Write-Host "Removed scheduled task '$taskName'."
    exit 0
}

if ($InstallStartup) {
    $action = New-ScheduledTaskAction `
        -Execute $powershell `
        -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$PSCommandPath`"" `
        -WorkingDirectory $repoRoot
    $trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
    $settings = New-ScheduledTaskSettingsSet `
        -ExecutionTimeLimit (New-TimeSpan -Days 1) `
        -RestartCount 3 `
        -RestartInterval (New-TimeSpan -Minutes 1)

    Register-ScheduledTask `
        -TaskName $taskName `
        -Action $action `
        -Trigger $trigger `
        -Settings $settings `
        -Description "Start the DeepFind web backend when the user signs in" `
        -Force | Out-Null

    Start-ScheduledTask -TaskName $taskName
    Write-Host "Registered '$taskName' to start DeepFind at sign-in."
    exit 0
}

if (-not (Test-Path $python -PathType Leaf)) {
    throw "DeepFind virtual environment not found. Run 'uv sync' in $repoRoot first."
}

New-Item -ItemType Directory -Force -Path $stateDir, $logDir | Out-Null

if (Test-Path $pidFile -PathType Leaf) {
    $existingPid = Get-Content $pidFile -ErrorAction SilentlyContinue
    $existingProcess = if ($existingPid -match "^\d+$") {
        Get-Process -Id $existingPid -ErrorAction SilentlyContinue
    }
    else {
        $null
    }
    if ($existingProcess) {
        if (-not $Restart) {
            Write-Host "DeepFind backend is already running with PID $existingPid."
            exit 0
        }
        Stop-Process -InputObject $existingProcess
        for ($attempt = 0; $attempt -lt 100; $attempt++) {
            if (-not (Get-Process -Id $existingPid -ErrorAction SilentlyContinue)) {
                break
            }
            Start-Sleep -Milliseconds 100
        }
        if (Get-Process -Id $existingPid -ErrorAction SilentlyContinue) {
            throw "DeepFind backend PID $existingPid did not stop."
        }
    }
    Remove-Item $pidFile -Force
}

try {
    $health = Invoke-RestMethod `
        -Uri "http://127.0.0.1:$port/api/health" `
        -TimeoutSec $healthTimeoutSec
    if ($health.ok -eq $true -and $health.service -eq "deepfind-web") {
        if ($Restart) {
            throw "DeepFind is responding on port $port, but its PID is not tracked in $pidFile. Stop that process explicitly before restarting."
        }
        Write-Host "DeepFind backend is already responding at http://127.0.0.1:$port."
        exit 0
    }
}
catch {
}

$backend = Start-Process `
    -FilePath $python `
    -ArgumentList "-m", "deepfind.web_api", "--host", $hostAddress, "--port", $port `
    -WorkingDirectory $repoRoot `
    -RedirectStandardOutput $stdoutLog `
    -RedirectStandardError $stderrLog `
    -WindowStyle Hidden `
    -PassThru
$backend.Id | Set-Content $pidFile

$deadline = [DateTime]::UtcNow.AddSeconds($startupTimeoutSec)
while ([DateTime]::UtcNow -lt $deadline) {
    try {
        $health = Invoke-RestMethod `
            -Uri "http://127.0.0.1:$port/api/health" `
            -TimeoutSec $healthTimeoutSec
        if ($health.ok -eq $true -and $health.service -eq "deepfind-web") {
            Write-Host "DeepFind backend started with PID $($backend.Id) at http://${hostAddress}:$port."
            Write-Host "Logs: $stdoutLog and $stderrLog"
            exit 0
        }
    }
    catch {
        $backend.Refresh()
        if ($backend.HasExited) {
            break
        }
    }
    Start-Sleep -Milliseconds 500
}

Remove-Item $pidFile -Force -ErrorAction SilentlyContinue
throw "DeepFind backend failed to start. Check $stdoutLog and $stderrLog."
