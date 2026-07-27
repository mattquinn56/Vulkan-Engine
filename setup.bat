@echo off
setlocal

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\setup.ps1" -Build -Config Release
if errorlevel 1 (
    echo.
    echo Setup failed. Review the message above, then run setup.bat again.
    pause
    exit /b 1
)
