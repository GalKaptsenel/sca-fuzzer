#!/usr/bin/env python3
"""Serve the AArch64 docs over HTTP(S), rebuilding the HTML from index.md on every
page load so the browser always shows the latest source.
Usage: serve_docs.py [port] [certfile]   (certfile -> serve over TLS; ports <1024 need sudo)."""
import http.server, socketserver, subprocess, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 80
CERT = sys.argv[2] if len(sys.argv) > 2 else None


class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/", "/index.html"):
            subprocess.run([sys.executable, os.path.join(HERE, "build_docs.py")], cwd=HERE)
        return super().do_GET()

    def log_message(self, *a):
        pass


os.chdir(HERE)
socketserver.TCPServer.allow_reuse_address = True
httpd = socketserver.TCPServer(("0.0.0.0", PORT), Handler)
scheme = "http"
if CERT:
    import ssl
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(CERT)
    httpd.socket = ctx.wrap_socket(httpd.socket, server_side=True)
    scheme = "https"
print(f"serving docs on {scheme}://0.0.0.0:{PORT}/ (rebuilds on each load)")
httpd.serve_forever()
