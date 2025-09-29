#!/usr/bin/python3
import os
import sys
import subprocess
import re
from utils import run_command, wait_for_port, PROJECT_ROOT, BUILD_DIR

PROGRAM_PATH = os.path.join(BUILD_DIR, "program")

def get_function_address(program_path, func_name):
    print(f"[*] Getting address of {func_name} in {program_path}...")
    result = subprocess.run(
        ["nm", program_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    func_pattern = re.compile(r"^\S+\s+T\s+" + re.escape(func_name) + r"$")
    for line in result.stdout.splitlines():
        if func_pattern.match(line):
            addr = line.split()[0]
            print(f"[+] Found {func_name} at {addr}")
            return addr
    print(f"[-] Could not find {func_name} function address.")
    return None

def main():
    if len(sys.argv) != 4:
        print("Usage: script.py <input_file> <testcase_file> <output_file>")
        sys.exit(1)

    input_file, testcase_file, output_file = sys.argv[1:4]

    os.makedirs(BUILD_DIR, exist_ok=True)

    # Step 1: Run loader_make.py
    run_command(f"{os.path.dirname(__file__)}/loader_make.py {input_file} {testcase_file}")

    # Step 2: Get function address
    func_address = get_function_address(PROGRAM_PATH, "func")
    if not func_address:
        sys.exit(1)

    # Step 3: Kill old lldb-server, start new one
    run_command("adb shell pkill -f lldb-server || true")
    run_command('adb shell "nohup /data/local/tmp/lldb-server platform --listen \\":1234\\" --server > /dev/null 2>&1 &"')

    # Step 4: Optionally wait for port
    # wait_for_port("localhost", 1234, timeout=10)

    # Step 5: Generate LLDB script
    lldb_commands = f"""platform select remote-android
platform connect connection://localhost:1234
platform settings -w /data/local/tmp
file {PROGRAM_PATH}
process launch --stop-at-entry
br set --address {func_address}
c
step_cfg_until_ret
quit
"""
    script_path = os.path.join(BUILD_DIR, "temp_lldb_script.txt")
    with open(script_path, "w") as f:
        f.write(lldb_commands)

    # Step 6: Run LLDB
    with open(output_file, "w") as outfile:
        run_command(f"lldb --batch --source {script_path}", stdout=outfile, stderr=subprocess.STDOUT)

    os.remove(script_path)

if __name__ == "__main__":
    main()

