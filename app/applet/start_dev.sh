#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT/backend"

# Ensure pip is installed if missing
if ! command -v pip3 >/dev/null 2>&1 && ! python3 -m pip --version >/dev/null 2>&1; then
  curl -sS https://bootstrap.pypa.io/get-pip.py | python3 - --break-system-packages 2>/dev/null || true
fi

# Ensure python packages are present if missing
if ! python3 -c 'import flask, flask_cors, plotly, pandas, numpy, pytz, yfinance' 2>/dev/null; then
  python3 -m pip install --break-system-packages flask flask-cors plotly pandas numpy pytz yfinance 2>/dev/null || true
fi

export KRONOS_REPO_PATH="${KRONOS_REPO_PATH:-$HOME/kronos_repo}"
if [ ! -f "$KRONOS_REPO_PATH/model.py" ]; then
  mkdir -p "$(dirname "$KRONOS_REPO_PATH")"
  git clone --depth 1 https://github.com/shiyu-coder/Kronos.git "$KRONOS_REPO_PATH" 2>/dev/null || true
fi

exec python3 nse_dashboard_enhanced.py
