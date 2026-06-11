from __future__ import annotations
from .models import OperandKind, MemAccess, MemRole, RegFile, ExtractionError
from .operands import _COND_CODES, _EXTEND_OPS

# Every extracted Instruction is run through this before it is emitted: a field with the wrong type,
# outside its domain, or inconsistent with another field is a parsing bug, so it loud-fails (recorded
# per-encoding) rather than reaching the output. No value is ever silently coerced or dropped.

_ESIZES = frozenset({0, 8, 16, 32, 64})
_REG_WIDTHS = {
    RegFile.GP: frozenset({0, 32, 64}),
    RegFile.SIMD: frozenset({0, 8, 16, 32, 64, 128}),
    RegFile.SVE_Z: frozenset({0}),
    RegFile.SVE_P: frozenset({0}),
}


def _require(ok, enc, msg):
    if not ok:
        raise ExtractionError(f"{enc}: malformed extraction: {msg}")


def _validate_operand(o, enc):
    _require(isinstance(o.width, int) and o.width >= 0, enc, f"{o.name!r} width {o.width!r} not int>=0")
    _require(isinstance(o.read, bool) and isinstance(o.write, bool), enc, f"{o.name!r} read/write not bool")

    for r in o.imm_ranges:
        _require(len(r) == 4 and all(isinstance(x, int) for x in r), enc, f"{o.name!r} imm_range {r!r} not 4 ints")
        esize, lo, hi, stride = r
        _require(esize in _ESIZES, enc, f"{o.name!r} imm_range esize {esize} invalid")
        _require(lo <= hi and stride >= 1, enc, f"{o.name!r} imm_range {r!r} lo>hi or stride<1")
    # esize 0 means "applies to any element size", so it can only be the sole entry
    _require(not (any(r[0] == 0 for r in o.imm_ranges) and len(o.imm_ranges) > 1),
             enc, f"{o.name!r} mixes an unconditional (esize 0) range with per-esize ranges")

    if o.reg_range:
        _require(len(o.reg_range) == 3 and all(isinstance(x, int) for x in o.reg_range),
                 enc, f"{o.name!r} reg_range {o.reg_range!r} not 3 ints")
        lo, hi, stride = o.reg_range
        _require(0 <= lo <= hi <= 31 and stride >= 1, enc, f"{o.name!r} reg_range {o.reg_range} outside [0,31]")

    _require(not o.sp_capable or o.reg_file is RegFile.GP,
             enc, f"{o.name!r} is sp_capable but not a GP register")
    if o.kind is OperandKind.LABEL:
        _require(o.read and not o.write, enc, f"label {o.name!r} must be read-only")
    if o.kind is OperandKind.REG:
        _require(o.reg_file is not None and o.reg_range and o.asl_index,
                 enc, f"register {o.name!r} missing file/range/asl_index")
        _require(o.read or o.write, enc, f"register {o.name!r} neither read nor written")
        _require(o.width in _REG_WIDTHS[o.reg_file],
                 enc, f"register {o.name!r} width {o.width} invalid for {o.reg_file.value}")
    elif o.kind is OperandKind.COND:
        _require(o.values and set(o.values) <= _COND_CODES, enc, f"cond {o.name!r} values {o.values}")
    elif o.kind is OperandKind.EXTEND:
        _require(o.values and set(o.values) <= _EXTEND_OPS, enc, f"extend {o.name!r} values {o.values}")
    elif o.kind is OperandKind.IMM:
        _require(o.imm_ranges or o.values, enc, f"immediate {o.name!r} has no value set")
        _require(not (o.imm_ranges and o.values), enc, f"immediate {o.name!r} has both a range and a value list")


def validate_instruction(i):
    enc = i.encoding_name
    _require(bool(i.name) and bool(i.category), enc, "empty name or category")
    _require(set(i.flags.written) <= set("NZCV") and set(i.flags.read) <= set("NZCV"),
             enc, "flags not a subset of NZCV")
    reg_vars = {o.asl_index for o in i.operands}
    for o in i.operands:
        _validate_operand(o, enc)
    for a, b in i.constraints:
        _require(a in reg_vars and b in reg_vars, enc, f"constraint ({a},{b}) names an unknown reg-var")

    # memory access and addressing operands must agree
    roles = {o.mem_role for o in i.operands}
    if i.mem_access is not MemAccess.NONE:
        addressed = MemRole.BASE in roles or any(o.kind is OperandKind.LABEL for o in i.operands)
        _require(addressed, enc, f"{i.mem_access.value} access but no base register or PC-relative label")
    else:
        _require(roles <= {MemRole.NONE}, enc, "non-memory instruction has an addressing-role operand")
