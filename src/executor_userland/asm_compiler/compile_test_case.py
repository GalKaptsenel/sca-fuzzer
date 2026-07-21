#!/usr/bin/env python3
"""Compile an AArch64 assembly test case into the raw flat binary the kernel executor loads.

The kernel module's test-case body is raw AArch64 machine code (the assembled, stripped `.text`, no
ELF wrapper). This utility turns a `.asm` file into exactly those bytes and reuses the Revizor
assembler path (Aarch64Generator.in_memory_assemble -> the asm_to_bytes helper = cross `as` +
`objcopy -O binary`), so it produces byte-identical output to what the fuzzer loads.

Load the result onto the device with, e.g.:
    executor_userland /dev/executor w test_case.bin
"""
import os
import sys
import argparse

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir))
sys.path.insert(0, _ROOT)

from src.aarch64.aarch64_generator import Aarch64Generator  # noqa: E402
from src.interfaces import PAGE_SIZE  # noqa: E402

MAX_TEST_CASE_SIZE = PAGE_SIZE   # kernel executor.h: MAX_TEST_CASE_SIZE = 1 * PAGESIZE


def compile_asm(asm: str) -> bytes:
    return Aarch64Generator.in_memory_assemble(asm)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile an AArch64 asm test case to a raw binary.")
    parser.add_argument("--input", help="Assembly file to compile (default: stdin)", required=False)
    parser.add_argument("--output", help="Path to the output raw binary", required=True)
    parser.add_argument("--print", help="Print the machine code as hex", action="store_true")
    args = parser.parse_args()

    asm = open(args.input).read() if args.input else sys.stdin.read()
    code = compile_asm(asm)

    with open(args.output, "wb") as f:
        f.write(code)
    print(f"[i] wrote {len(code)} bytes of machine code to {args.output}")
    if len(code) > MAX_TEST_CASE_SIZE:
        print(f"[!] warning: {len(code)} bytes exceeds MAX_TEST_CASE_SIZE ({MAX_TEST_CASE_SIZE}); "
              "the kernel will reject it")
    if args.print:
        print(code.hex())


if __name__ == "__main__":
    main()
