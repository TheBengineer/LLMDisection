@echo off
setlocal

:: =============================================================================
:: Qwen2.5-0.5B Visual Step-Through Explorer — Windows Launcher
:: =============================================================================

title Qwen2.5-0.5B Visual Step-Through Explorer

echo ==============================================
echo   Qwen2.5-0.5B Visual Step-Through Explorer
echo ==============================================
echo.

:: --- Locate Python ---
set "PYTHON_CMD=python"
where python >nul 2>&1
if errorlevel 1 (
    where python3 >nul 2>&1
    if errorlevel 1 (
        echo [ERROR] Python not found. Install Python 3.10+ and try again.
        pause
        exit /b 1
    )
    set "PYTHON_CMD=python3"
)

echo [*] Using: %PYTHON_CMD%

:: --- Check version ---
setlocal enabledelayedexpansion
for /f "tokens=2 delims=." %%v in ('%PYTHON_CMD% --version 2^>^&1') do (
    set "PY_MAJOR=%%v"
    goto :version_checked
)
:version_checked
if "!PY_MAJOR!" LSS "10" (
    echo [ERROR] Python 3.10+ required, found Python 3.!PY_MAJOR!
    pause
    exit /b 1
)
endlocal

:: --- Install missing dependencies ---
echo [*] Checking dependencies...
%PYTHON_CMD% -c "import torch, transformers, gradio, plotly, numpy" >nul 2>&1
if errorlevel 1 (
    echo [*] Installing required packages...
    %PYTHON_CMD% -m pip install --upgrade pip -q
    %PYTHON_CMD% -m pip install torch transformers gradio plotly numpy -q
    if errorlevel 1 (
        echo [ERROR] pip install failed.
        pause
        exit /b 1
    )
    echo [*] Dependencies installed.
)

echo.
echo [*] Starting server...
echo [*] Open http://localhost:7860 in your browser
echo.

:: --- Launch app and open browser ---
start "" http://localhost:7860
%PYTHON_CMD% app.py

echo.
pause