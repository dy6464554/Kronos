"""AI Studio launcher for the Upstox Kronos terminal."""
#!/bin/bash
set -e
cd "$(dirname "$0")/backend"
python3 -m pip install -r requirements.txt --break-system-packages 2>/dev/null || python3 -m pip install -r requirements.txt
exec python3 nse_dashboard_enhanced.py
