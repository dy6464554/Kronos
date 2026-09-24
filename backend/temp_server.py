from http.server import BaseHTTPRequestHandler, HTTPServer
class MyHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        self.wfile.write(b"<html><head><meta http-equiv='refresh' content='5'></head><body><h1>Installing Machine Learning Dependencies...</h1><p>Please wait (this may take a few minutes for PyTorch)...</p></body></html>")
httpd = HTTPServer(('', 3000), MyHandler)
httpd.serve_forever()
