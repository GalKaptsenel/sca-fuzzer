#!/bin/python
import json
import random
import argparse
import struct
from typing import Dict, Tuple

# Constants
KB = 1024
MAIN_REGION_SIZE = 4 * KB
FAULTY_REGION_SIZE = 4 * KB
REG_INITIALIZATION_REGION_SIZE_ALIGNED = 4 * KB
TOTAL_SIZE = MAIN_REGION_SIZE + FAULTY_REGION_SIZE + REG_INITIALIZATION_REGION_SIZE_ALIGNED

REGISTER_NAMES = ["x0", "x1", "x2", "x3", "x4", "x5", "flags", "sp"]


def parse_memory_keys(mem_dict):
    parsed = {}
    for k, v in mem_dict.items():
        try:
            parsed[int(k, 0)] = v
        except ValueError:
            raise ValueError(f"Invalid memory key format: {k}")
    return parsed

def fill_memory_region(buffer: bytearray, offset: int, size: int, user_region_raw: Dict[str, int]):
    """
    user_region_raw is a dict with string keys: "0" or "0x10" (memory offsets) and integer byte values
    """
    def normalize_byte(val):
        return val & 0xFF

    def random_byte():
        return random.getrandbits(8)

    user_region = parse_memory_keys(user_region_raw)

    for i in range(size):
        raw_val = user_region.get(i, random_byte())
        buffer[offset + i] = normalize_byte(raw_val)

def fill_registers(buffer, offset, user_regs):
    def to_u64(val):
        return val & 0xFFFFFFFFFFFFFFFF

    def random_u64():
        return random.getrandbits(64)


    reg_values = {}
    for reg in REGISTER_NAMES:
        raw_val = user_regs.get(reg, random_u64())
        val = to_u64(raw_val)
        struct.pack_into("<Q", buffer, offset, val)
        reg_values[reg] = val
        offset += 8
    return reg_values

def generate_binary_input(user_input=None):
    user_input = user_input or {}

    buffer = bytearray(TOTAL_SIZE)

    user_main_raw = user_input.get("memory", {}).get("main_region", {})
    fill_memory_region(buffer, offset=0, size=MAIN_REGION_SIZE, user_region_raw=user_main_raw)

    user_faulty_raw = user_input.get("memory", {}).get("faulty_region", {})
    fill_memory_region(buffer, offset=MAIN_REGION_SIZE, size=FAULTY_REGION_SIZE, user_region_raw=user_faulty_raw)

    reg_offset = MAIN_REGION_SIZE + FAULTY_REGION_SIZE
    user_regs = user_input.get("registers", {})
    reg_values = fill_registers(buffer, offset=reg_offset, user_regs=user_regs)

    return buffer, reg_values

def hexdump(data, offset_start=0, width=16):
    lines = []
    for i in range(0, len(data), width):
        chunk = data[i:i + width]
        hex_part = " ".join(f"{b:02x}" for b in chunk)
        ascii_part = "".join(chr(b) if 32 <= b <= 126 else '.' for b in chunk)
        lines.append(f"{offset_start + i:08x}  {hex_part:<{width*3}}  {ascii_part}")
    return "\n".join(lines)

def write_pretty_output(filepath, binary_data, reg_values):
    lines = []

    lines.append("Registers:")
    for reg in REGISTER_NAMES:
        val = reg_values[reg]
        lines.append(f"  {reg:<6} = 0x{val:016x}")
    lines.append("")

    lines.append("<------ START OF MAIN REGION ------>")
    lines.append(hexdump(binary_data[0:MAIN_REGION_SIZE], offset_start=0))
    lines.append("")

    lines.append("<------ START OF FAULTY REGION ------>")
    lines.append(hexdump(binary_data[MAIN_REGION_SIZE:MAIN_REGION_SIZE + FAULTY_REGION_SIZE],
                         offset_start=MAIN_REGION_SIZE))
    lines.append("")

    lines.append("<------ START OF REGISTER REGION ------>")
    lines.append(hexdump(binary_data[MAIN_REGION_SIZE + FAULTY_REGION_SIZE:],
                         offset_start=MAIN_REGION_SIZE + FAULTY_REGION_SIZE))
    lines.append("")

    with open(filepath, "w") as f:
        f.write("\n".join(lines))

    print(f"[i] Hexdump written to {filepath}")

def main():
    parser = argparse.ArgumentParser(description="Generate binary input_t structure.")
    parser.add_argument("--input", help="Optional JSON file with memory/register overrides", required=False)
    parser.add_argument("--output", help="Path to output binary file", required=True)
    parser.add_argument("--print", help="Also write a hexdump alongside output", action="store_true")
    args = parser.parse_args()

    user_input = {}
    if args.input:
        with open(args.input, "r") as f:
            user_input = json.load(f)

    binary_data, reg_values = generate_binary_input(user_input)

    with open(args.output, "wb") as f:
        f.write(binary_data)

    # Optionally write hexdump
    if args.print:
        hex_text = hexdump(binary_data)
        txt_output = args.output + ".txt"
        write_pretty_output(txt_output, binary_data, reg_values)

if __name__ == "__main__":
    main()

