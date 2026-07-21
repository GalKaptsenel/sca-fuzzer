#!/usr/bin/env python3
"""Generate a REIF input file for /dev/executor from a JSON spec.

This is the refactor of the old ``generate_input_binary.py``, which emitted the retired flat
``input_t`` dump (main[4K] | faulty[4K] | reg[4K]). The JSON spec is unchanged, but the output is now
a ``.reif`` file: the REIF format documented in docs/reif_input_format.md and defined by the kernel
header src/aarch64/executor/userapi/executor_input_format.h. The file is executor-ready — write it
straight into /dev/executor (e.g. ``executor_userland /dev/executor w input.reif``).

All REIF byte-layout logic is reused from the Revizor encoder
(src/aarch64/aarch64_executor_input_encoder.build_input_init); this tool only turns the JSON into the
three raw section payloads (MEMORY_MAIN, MEMORY_FAULTY, GPR) and hands them over.

JSON spec (all fields optional; anything unspecified is randomised):
  {
    "registers": { "x0..x5": int, "flags": int, "sp": int },   # flags/sp written verbatim
    "memory": {
      "main_region":   { "<byte-offset>": <byte>, ... },
      "faulty_region": { "<byte-offset>": <byte>, ... }
    }
  }
"""
import os
import sys
import json
import struct
import random
import argparse
from typing import Dict, Tuple

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir))
sys.path.insert(0, _ROOT)

from src.aarch64.aarch64_executor_input_encoder import build_input_init  # noqa: E402
from src.interfaces import MAIN_AREA_SIZE, FAULTY_AREA_SIZE, GPR_SUBREGION_SIZE  # noqa: E402

REGISTER_NAMES = ("x0", "x1", "x2", "x3", "x4", "x5", "flags", "sp")
GPR_WORDS = GPR_SUBREGION_SIZE // 8


def _fill_region(size: int, overrides_raw: Dict[str, int], rng: random.Random) -> bytes:
    buffer = bytearray(rng.randbytes(size))
    for key, value in overrides_raw.items():
        try:
            offset = int(key, 0)
        except ValueError:
            raise ValueError(f"Invalid memory offset key: {key!r}")
        if not 0 <= offset < size:
            raise ValueError(f"Memory offset {offset} out of range [0,{size})")
        buffer[offset] = value & 0xFF
    return bytes(buffer)


def _gpr_section(user_regs: Dict[str, int], rng: random.Random) -> Tuple[bytes, Dict[str, int]]:
    values = {}
    words = []
    for name in REGISTER_NAMES:
        value = user_regs.get(name, rng.getrandbits(64)) & 0xFFFFFFFFFFFFFFFF
        values[name] = value
        words.append(value)
    return struct.pack(f"<{GPR_WORDS}Q", *words), values


def generate_reif(spec: Dict, rng: random.Random) -> Tuple[bytes, Dict[str, int]]:
    memory = spec.get("memory", {})
    main = _fill_region(MAIN_AREA_SIZE, memory.get("main_region", {}), rng)
    faulty = _fill_region(FAULTY_AREA_SIZE, memory.get("faulty_region", {}), rng)
    gpr, reg_values = _gpr_section(spec.get("registers", {}), rng)
    return build_input_init(main, faulty, gpr), reg_values


def _write_pretty(path: str, reg_values: Dict[str, int]) -> None:
    lines = ["Registers:"]
    for name in REGISTER_NAMES:
        lines.append(f"  {name:<6} = 0x{reg_values[name]:016x}")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[i] register summary written to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a REIF input file from a JSON spec.")
    parser.add_argument("--input", help="JSON file with memory/register overrides", required=False)
    parser.add_argument("--output", help="Path to the output .reif file", required=True)
    parser.add_argument("--seed", help="Seed the RNG for reproducible fills", type=int, required=False)
    parser.add_argument("--print", help="Also write a register summary alongside output",
                        action="store_true")
    args = parser.parse_args()

    spec = {}
    if args.input:
        with open(args.input) as f:
            spec = json.load(f)

    rng = random.Random(args.seed)
    blob, reg_values = generate_reif(spec, rng)

    with open(args.output, "wb") as f:
        f.write(blob)
    print(f"[i] wrote {len(blob)} bytes of REIF to {args.output}")

    if args.print:
        _write_pretty(args.output + ".txt", reg_values)


if __name__ == "__main__":
    main()
