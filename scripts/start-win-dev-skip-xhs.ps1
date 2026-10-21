$ErrorActionPreference = 'Stop'

Add-Type @'
using System;
using System.Runtime.InteropServices;

public static class DeepFindConsoleWindow
{
    [DllImport("kernel32.dll")]
    public static extern IntPtr GetConsoleWindow();

    [DllImport("user32.dll")]
    public static extern bool ShowWindow(IntPtr hWnd, int nCmdShow);
}
'@

[DeepFindConsoleWindow]::ShowWindow(
    [DeepFindConsoleWindow]::GetConsoleWindow(),
    0
) | Out-Null

& "$PSScriptRoot\deploy_win.ps1" -SkipXhsLogin -HideDesktopConsole @args
