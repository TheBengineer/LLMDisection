#!/usr/bin/env bash
#
# run.sh — Launch the Qwen2.5-0.5B Visual Step-Through Explorer
#          and open the browser automatically.
#
# Usage:
#   ./run.sh              # use default port 7860
#   ./run.sh 8080         # use a custom port
#

set -euo pipefail

PORT="${1:-7860}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=============================================="
echo "  Qwen2.5-0.5B Visual Step-Through Explorer"
echo "=============================================="
echo ""

# ── Python environment ─────────────────────────────
# Prefer a virtual environment if one exists; otherwise
# fall back to the system Python.
if [ -d venv ]; then
    echo "[*] Activating virtual environment (./venv) …"
    source venv/bin/activate
elif [ -d .venv ]; then
    echo "[*] Activating virtual environment (./.venv) …"
    source .venv/bin/activate
fi

# ── Dependencies (one-time) ────────────────────────
if ! python3 -c "import torch, transformers, gradio, plotly" 2>/dev/null; then
    echo "[*] Installing dependencies …"
    pip install --quiet torch transformers gradio plotly
fi

# ── Launch ─────────────────────────────────────────
echo "[*] Starting server on port ${PORT} …"
echo "[*] Open http://localhost:${PORT} in your browser."
echo ""

# Open browser after a short delay (server may take a moment)
( sleep 3 && xdg-open "http://localhost:${PORT}" 2>/dev/null ) &

python3 app.py --port "$PORT"
