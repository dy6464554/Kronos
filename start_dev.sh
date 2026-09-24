#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT/backend"
python3 -m pip install -r requirements.txt --break-system-packages 2>/dev/null || python3 -m pip install -r requirements.txt
# Use the real Kronos package when it is not mounted by the runtime.
if [ ! -f "${KRONOS_REPO_PATH:-$HOME/kronos_repo}/model.py" ]; then
  export KRONOS_REPO_PATH="${TMPDIR:-/tmp}/kronos_repo"
  if [ ! -f "$KRONOS_REPO_PATH/model.py" ]; then
    git clone --depth 1 https://github.com/shiyu-coder/Kronos.git "$KRONOS_REPO_PATH"
  fi
fi
exec python3 nse_dashboard_enhanced.py
