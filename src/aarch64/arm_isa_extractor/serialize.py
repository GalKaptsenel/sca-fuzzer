from __future__ import annotations
import json
from .models import Instruction


def operand_dict(o) -> dict:
    return {
        "name": o.name, "kind": o.kind.value, "read": o.read, "write": o.write,
        "width": o.width, "signed": o.signed, "values": list(o.values),
        "imm_ranges": [list(r) for r in o.imm_ranges],
        "reg_range": list(o.reg_range) if o.reg_range else None,
        "reg_file": o.reg_file.value if o.reg_file else None,
        "asl_index": o.asl_index, "sp_capable": o.sp_capable,
        "mem_role": o.mem_role.value,
    }


def instruction_dict(i: Instruction) -> dict:
    return {
        "name": i.name, "iclass_id": i.iclass_id, "category": i.category,
        "encoding_name": i.encoding_name, "asm_template": i.asm_template,
        "control_flow": i.control_flow, "mem_access": i.mem_access.value,
        "flags_written": sorted(i.flags.written), "flags_read": sorted(i.flags.read),
        "operands": [operand_dict(o) for o in i.operands],
        "constraints": [list(c) for c in i.constraints],
    }


def write_json(instructions, path) -> None:
    json.dump([instruction_dict(i) for i in instructions], open(path, "w"), indent=1, sort_keys=True)
