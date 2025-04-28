#!/usr/bin/python
import subprocess
import sys
import time
import os
import socket
import re

def wait_for_port(host, port, timeout=10):
    print(f"[*] Waiting for {host}:{port} to become available...")
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            with socket.create_connection((host, port), timeout=1):
                print(f"[*] Port {port} is now open.")
                return True
        except (ConnectionRefusedError, socket.timeout):
            time.sleep(0.5)
    raise TimeoutError(f"Timeout: Could not connect to {host}:{port} within {timeout} seconds")


def get_function_address(program_path, func_name):
    # Run nm command to get function addresses
    print(f"[*] Getting address of {func_name} function in {program_path}...")
    result = subprocess.run(
        ["nm", program_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    
    # Search for function name in the output
    func_address = None
    func_pattern = re.compile(r"^\S+\s+T\s+" + re.escape(func_name) + r"$")

    for line in result.stdout.splitlines():
        if func_pattern.match(line):
            func_address = line.split()[0]  # First column is address
            print(f"[+] Found {func_name} at address: {func_address}")
            break

    if not func_address:
        print(f"[-] Could not find {func_name} function address.")
    
    return func_address

def main():
    if len(sys.argv) != 4:
        print("Usage: script.py <input_file> <testcase_file> <output_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    testcase_file = sys.argv[2]
    output_file = sys.argv[3]

    # Step 1: Run loader_make.py
    print("[*] Running loader_make.py...")
    subprocess.run([f"{os.path.dirname(os.path.realpath(__file__))}/loader_make.py", input_file, testcase_file], check=True)


    # Step 2: Get address of function using nm
    program_path = "program"
    func_name = "func"
    func_address = get_function_address(program_path, func_name)
    
    # If no function address is found, stop further execution
    if not func_address:
        print(f"Error: Function {func_name} not found. Exiting.")
        sys.exit(1)


    # Step 3: Kill previous lldb-server instances on device
    print("[*] Killing any existing lldb-server processes on device...")
    subprocess.run(["adb", "shell", "pkill", "-f", "lldb-server"], stderr=subprocess.DEVNULL)

    # Step 4: Start lldb-server in background on device
    print("[*] Starting lldb-server on Android device in background...")
    subprocess.run(
        'adb shell "nohup /data/local/tmp/lldb-server platform --listen \"*:1234\" --server > /dev/null 2>&1 &"',
        shell=True,
        check=True
    )

    # Step 5: Wait until the port is ready
    #wait_for_port("localhost", 1234, timeout=10)

    # Step 5: Create LLDB command script
    lldb_commands = f"""platform select remote-android
platform connect connection://localhost:1234
platform settings -w /data/local/tmp
file {program_path}
process launch --stop-at-entry
br set --address {func_address}
c
step_cfg_until_ret
quit
""" 

    lldb_script_path = "temp_lldb_script.txt"
    with open(lldb_script_path, "w") as f:
        f.write(lldb_commands)

    print("[*] Running LLDB with script...")
    with open(output_file, "w") as outfile:
        subprocess.run(["lldb", "--batch","--source", lldb_script_path], stdout=outfile, stderr=subprocess.STDOUT)

    # Cleanup
    if os.path.exists(lldb_script_path):
        os.remove(lldb_script_path)
        print("[*] Cleaned up temporary LLDB script.")

if __name__ == "__main__":
    main()
