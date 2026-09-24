#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT/backend"
python3 -m pip install -r requirements.txt --break-system-packages 2>/dev/null || python3 -m pip install -r requirements.txt
export KRONOS_REPO_PATH="${KRONOS_REPO_PATH:-$HOME/kronos_repo}"
if [ ! -f "$KRONOS_REPO_PATH/model.py" ]; then
  mkdir -p "$(dirname "$KRONOS_REPO_PATH")"
  git clone --depth 1 https://github.com/shiyu-coder/Kronos.git "$KRONOS_REPO_PATH" || true
fi
exec python3 nse_dashboard_enhanced.py
