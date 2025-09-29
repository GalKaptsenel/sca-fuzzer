import os
import subprocess
import socket
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
BUILD_DIR = os.path.join(PROJECT_ROOT, "build/userland")
INCLUDE_DIR = os.path.join(PROJECT_ROOT, "include")
THIRD_PARTY_DIR = os.path.join(PROJECT_ROOT, "third_party")

def run_command(command, **kwargs):
    print(f"🏃 Running: {command}")
    subprocess.run(command, shell=True, check=True, **kwargs)

def wait_for_port(host, port, timeout=10):
    print(f"[*] Waiting for {host}:{port} to become available...")
    start = time.time()
    while time.time() - start < timeout:
        try:
            with socket.create_connection((host, port), timeout=1):
                print(f"[+] Port {port} is now open.")
                return True
        except (ConnectionRefusedError, socket.timeout):
            time.sleep(0.5)
    raise TimeoutError(f"Timeout: Could not connect to {host}:{port} within {timeout}s")

