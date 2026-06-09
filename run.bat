@echo off
REM run.bat — Launch the Qwen2.5-0.5B Visual Step-Through Explorer
REM            and open the browser automatically.
REM
REM Usage:
REM   run.bat           use default port 7860
REM   run.bat 8080      use a custom port

setlocal enabledelayedexpansion

set PORT=%1
if "%PORT%"=="" set PORT=7860

echo ==============================================
echo   Qwen2.5-0.5B Visual Step-Through Explorer
echo ==============================================
echo.

REM ── Python environment ──────────────────────────
if exist venv\Scripts\activate.bat (
    echo [*] Activating virtual environment (venv) ...
    call venv\Scripts\activate.bat
) else if exist .venv\Scripts\activate.bat (
    echo [*] Activating virtual environment (.venv) ...
    call .venv\Scripts\activate.bat
)

REM ── Dependencies (one-time) ─────────────────────
python -c "import torch, transformers, gradio, plotly" 2>nul
if errorlevel 1 (
    echo [*] Installing dependencies ...
    pip install --quiet torch transformers gradio plotly
)

REM ── Launch ──────────────────────────────────────
echo [*] Starting server on port %PORT% ...
echo [*] Open http://localhost:%PORT% in your browser.
echo.

REM Open browser after a short delay
timeout /t 3 /nobreak >nul
start http://localhost:%PORT%

python app.py --port %PORT%

endlocal
