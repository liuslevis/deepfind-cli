@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0start-deepfind-cli.ps1" %*
exit /b %ERRORLEVEL%
