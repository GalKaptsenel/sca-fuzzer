#!/usr/bin/env python3
"""Convert a saved input into the byte format /dev/executor accepts.

The fuzzer stores an input's NZCV flags in register slot 6 using the per-flag
encoding (one byte per flag, bit 0 = flag value — see NZCVScheme); that is the
canonical, replayable form. The kernel executor instead expects that slot to
hold a real PSTATE value (flags in bits 31:28). The Python executor does this
conversion automatically when it writes an input, but manual reproduction via
executor_userland bypasses it — hence this tool.

It rewrites only the flags slot, byte-for-byte identical everywhere else, so the
result can be written straight to /dev/executor (or via executor_userland w).

Usage: to_executor_input.py <saved_input.bin> <executor_ready.bin>
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from src.aarch64.aarch64_input_layout import _reconstruct_pstate, REGISTER_REGION_OFFSET


def convert(raw: bytes) -> bytes:
    data = bytearray(raw)
    if len(data) < REGISTER_REGION_OFFSET + 64:
        raise ValueError(
            f"input too short ({len(data)} bytes); need at least "
            f"{REGISTER_REGION_OFFSET + 64} to hold the register region")
    # Reconstruct the flags slot in place; every other byte is left untouched.
    _reconstruct_pstate(memoryview(data)[REGISTER_REGION_OFFSET:].cast('Q'))
    return bytes(data)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="saved input (NZCVScheme per-flag flags encoding)")
    parser.add_argument("output", help="path to write the executor-ready input")
    args = parser.parse_args()

    with open(args.input, "rb") as f:
        converted = convert(f.read())
    with open(args.output, "wb") as f:
        f.write(converted)
    print(f"[i] wrote {len(converted)} bytes to {args.output} (flags slot -> PSTATE)")


if __name__ == "__main__":
    main()
