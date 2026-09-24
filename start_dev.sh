#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

# Keep the reference dashboard UI verbatim, changing only provider labels and
# endpoint names. The Python implementation remains local and uses Upstox.
cp index.html backend/templates/nse_dashboard_enhanced.html
python3 - <<'PY'
from pathlib import Path
p = Path('backend/templates/nse_dashboard_enhanced.html')
s = p.read_text()
replacements = {
    'Kite Live': 'Upstox Live', 'Kite Connect': 'Upstox Connect',
    'kite-login': 'upstox-login', 'kite-connect': 'upstox-connect',
    'checkKiteStatus': 'checkUpstoxStatus', 'updateKiteUI': 'updateUpstoxUI',
    'connectKite': 'connectUpstox', 'kite-lbl': 'upstox-lbl',
    'src-kit': 'src-upstox', 'on-k': 'on-upstox',
    'kite_connected': 'upstox_connected', 'KITE LIVE': 'UPSTOX LIVE',
    "src==='kite'": "src==='upstox'", "src==='yfinance'": "src==='upstox'",
    "'yfinance'": "'upstox'", "'kite'": "'upstox'",
    'yfinance': 'Upstox', 'Kite': 'Upstox', 'kite': 'upstox',
}
for old, new in replacements.items(): s = s.replace(old, new)
p.write_text(s)
PY

# The applet launcher is intentionally a valid shell script (the prior version
# accidentally contained a Python docstring before the shebang).
cd backend
python3 -m pip install -r requirements.txt --break-system-packages 2>/dev/null || python3 -m pip install -r requirements.txt
exec python3 nse_dashboard_enhanced.py
