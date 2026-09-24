#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT/backend"
python3 -m pip install -r requirements.txt --break-system-packages 2>/dev/null || python3 -m pip install -r requirements.txt
if [ -z "${KRONOS_REPO_PATH:-}" ]; then
  export KRONOS_REPO_PATH="${HOME}/kronos_repo"
fi
if [ ! -f "$KRONOS_REPO_PATH/model.py" ]; then
  mkdir -p "$(dirname "$KRONOS_REPO_PATH")"
  if [ ! -d "$KRONOS_REPO_PATH" ]; then
    git clone --depth 1 https://github.com/shiyu-coder/Kronos.git "$KRONOS_REPO_PATH" || true
  fi
fi
exec python3 nse_dashboard_enhanced.py
