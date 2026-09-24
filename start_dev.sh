#!/bin/bash
echo "Starting Dev Server Wrapper"
cd backend

echo "Checking python dependencies..."
python3 -c 'import torch, pandas, flask, plotly, yfinance, einops' 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Dependencies missing, starting temp server..."
    # Start a temporary server to bind port 3000 immediately so AI Studio doesn't timeout
    cat << 'PYEOF' > temp_server.py
from http.server import BaseHTTPRequestHandler, HTTPServer
class MyHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        self.end_headers()
        self.wfile.write(b"<html><head><meta http-equiv='refresh' content='5'></head><body><h1>Installing Machine Learning Dependencies...</h1><p>Please wait (this may take a few minutes for PyTorch)...</p></body></html>")
httpd = HTTPServer(('', 3000), MyHandler)
httpd.serve_forever()
PYEOF
    python3 temp_server.py &
    TEMP_PID=$!
    
    echo "Installing pip and dependencies..."
    dpkg --configure -a --force-confdef --force-confold
    DEBIAN_FRONTEND=noninteractive apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get install -y -o Dpkg::Options::="--force-confdef" -o Dpkg::Options::="--force-confold" python3-pip
    python3 -m pip install numpy pandas flask flask_cors plotly yfinance einops huggingface_hub safetensors --break-system-packages
    python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu --break-system-packages --default-timeout=1000

    # Stop temporary server
    kill $TEMP_PID 2>/dev/null
    sleep 1
fi

# Start actual app
export KRONOS_REPO_PATH=~/kronos_repo
python3 nse_dashboard_enhanced.py
