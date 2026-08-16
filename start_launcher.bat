@echo off
REM ---------------------------------------------------------
REM Activates whisper-env, runs launcher_Test4.py in the
REM background, waits until it's actually ready, then opens
REM the Launcher page in your default browser automatically.
REM
REM Edit the paths below if your folders are different.
REM ---------------------------------------------------------

set VENV_PATH=C:\Users\hi\whisper-env
set SCRIPT_PATH=C:\Users\baif-translation-pipeline-lite\Front_end\launcher.py
set LAUNCHER_URL=http://127.0.0.1:5050/
set LAUNCHER_PORT=5050

call "%VENV_PATH%\Scripts\activate.bat"

REM Start the launcher in a separate, minimized window so this
REM window can move on to waiting/opening the browser.
start "Launcher" /min python "%SCRIPT_PATH%"

REM Poll until port 5050 is actually accepting connections
REM (up to ~30 seconds) before opening the browser.
powershell -NoProfile -Command ^
  "$ready = $false; for ($i = 0; $i -lt 30; $i++) { try { $c = New-Object System.Net.Sockets.TcpClient('127.0.0.1', %LAUNCHER_PORT%); $c.Close(); $ready = $true; break } catch { Start-Sleep -Seconds 1 } }; if (-not $ready) { exit 1 }"

if errorlevel 1 (
    echo Launcher did not start within the expected time.
    pause
    exit /b 1
)

start "" "%LAUNCHER_URL%"
