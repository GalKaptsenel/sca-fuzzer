"""
File: AArch64 instruction disassembly helpers (capstone-based).
  - Decode a 32-bit encoding to text, read/written operands, branch class
"""
from typing import List, Tuple

from capstone import Cs, CS_ARCH_ARM64, CS_MODE_ARM, CS_AC_READ, CS_AC_WRITE
from capstone.arm64 import (ARM64_OP_REG, ARM64_OP_MEM, ARM64_OP_IMM,
                            ARM64_CC_INVALID, ARM64_CC_EQ, ARM64_CC_NE,
                            ARM64_CC_HS, ARM64_CC_LO, ARM64_CC_MI, ARM64_CC_PL,
                            ARM64_CC_VS, ARM64_CC_VC, ARM64_CC_HI, ARM64_CC_LS,
                            ARM64_CC_GE, ARM64_CC_LT, ARM64_CC_GT, ARM64_CC_LE,
                            ARM64_CC_AL, ARM64_CC_NV)

_CAPSTONE = Cs(CS_ARCH_ARM64, CS_MODE_ARM)
_CAPSTONE.detail = True

# Capstone 5.0.x leaves op.access empty for these MTE ops, so their register roles must be filled in
# by position: the first register operand is the destination, any further register operands are
# sources (memory bases are recovered separately). SUBPS additionally writes NZCV.
_MTE_FIRST_REG_DEST = frozenset({"addg", "subg", "irg", "gmi", "subp", "subps", "ldg"})

# Capstone 5.0.x under-reports the FEAT_FlagM/FlagM2 flag-manipulation ops: it exposes neither the
# NZCV bits they read nor the ones they write. Model each precisely as (reads, writes) over PSTATE
# flags: a flag the op PRESERVES must not be listed as written (that would silently drop live taint on
# it), and a flag it READS must be listed (a missed read under-taints). Any register operands are
# added as sources separately. RMIF's write set depends on its mask immediate (see below); it is not
# in this table.
_FLAG_OP_NZCV = {
    "setf8":  (set(),                {"N", "Z", "V"}),        # N,Z,V from Xn; C preserved
    "setf16": (set(),                {"N", "Z", "V"}),        # N,Z,V from Xn; C preserved
    "cfinv":  ({"C"},                {"C"}),                  # C := NOT C
    "axflag": ({"Z", "C", "V"},      {"N", "Z", "C", "V"}),   # ARM -> alternate FP flag format
    "xaflag": ({"C", "Z"},           {"N", "Z", "C", "V"}),   # alternate -> ARM FP flag format
}

# RMIF writes only the NZCV bits selected by its 4-bit mask immediate and preserves the rest:
# bit 3 -> N, bit 2 -> Z, bit 1 -> C, bit 0 -> V (ARM DDI 0487, RMIF).
_RMIF_MASK_FLAGS = (("N", 8), ("Z", 4), ("C", 2), ("V", 1))


def _rmif_written_flags(insn) -> set:
    """The NZCV bits RMIF actually writes, decoded from its mask operand (RMIF's last immediate)."""
    masks = [op.imm for op in insn.operands if op.type == ARM64_OP_IMM]
    if not masks:
        raise ValueError(f"RMIF without a mask immediate: 0x{insn.bytes.hex()}")
    mask = masks[-1]
    return {flag for flag, bit in _RMIF_MASK_FLAGS if mask & bit}


def decode_reg_accesses(encoding: int, pc: int) -> Tuple[List[str], List[str]]:
    FLAG_BITS = {"N", "Z", "C", "V"}

    def cc_to_read_flags(cc: int):
        cond_map = {
            ARM64_CC_EQ: {"Z"},
            ARM64_CC_NE: {"Z"},
            ARM64_CC_HS: {"C"},
            ARM64_CC_LO: {"C"},
            ARM64_CC_MI: {"N"},
            ARM64_CC_PL: {"N"},
            ARM64_CC_VS: {"V"},
            ARM64_CC_VC: {"V"},
            ARM64_CC_HI: {"C", "Z"},
            ARM64_CC_LS: {"C", "Z"},
            ARM64_CC_GE: {"N", "V"},
            ARM64_CC_LT: {"N", "V"},
            ARM64_CC_GT: {"Z", "N", "V"},
            ARM64_CC_LE: {"Z", "N", "V"},
            ARM64_CC_AL: set(),
            ARM64_CC_NV: set(),
        }
        # Unknown condition code: conservatively assume all flags are read — a safe
        # over-approximation for taint (never under-reports a branch's flag inputs).
        return cond_map.get(cc, {"N", "Z", "C", "V"})

    code_bytes = encoding.to_bytes(4, byteorder="little")
    insns = list(_CAPSTONE.disasm(code_bytes, pc))
    if len(insns) != 1:
        raise ValueError(f"expected exactly one instruction decoding 0x{encoding:08x} "
                         f"at pc 0x{pc:x}, got {len(insns)}")
    insn = insns[0]

    dest = set()
    src = set()
    if insn.update_flags:
        dest |= FLAG_BITS
    # ARM64_CC_INVALID means unconditional; only add flag reads for real conditions.
    if insn.cc is not None and insn.cc != ARM64_CC_INVALID:
        src |= cc_to_read_flags(insn.cc)

    mnemonic = insn.mnemonic.lower()
    reg_roles_fixed = mnemonic in _MTE_FIRST_REG_DEST or mnemonic in ("rmif", "setf8", "setf16",
                                                                      "pacga")

    for op in insn.operands:
        if op.type == ARM64_OP_REG:
            reg = insn.reg_name(op.reg)
            if op.access & CS_AC_WRITE:
                dest.add(reg)
            if op.access & CS_AC_READ:
                src.add(reg)
            # Access empty and no explicit role: over-approximate as a read (taint-safe; a spurious
            # write would not be).
            if not (op.access & (CS_AC_READ | CS_AC_WRITE)) and not reg_roles_fixed:
                src.add(reg)
        elif op.type == ARM64_OP_MEM:
            if op.mem.base != 0:
                base_reg = insn.reg_name(op.mem.base)
                src.add(base_reg)
                # pre/post-index writeback also UPDATES the base register; Capstone leaves it out of
                # the operand access flags, so add it explicitly.
                if getattr(insn, "writeback", False):
                    dest.add(base_reg)
            if op.mem.index != 0:
                src.add(insn.reg_name(op.mem.index))

    # Capstone 5.0.x under-reports these: the FEAT_FlagM/FlagM2 flag ops (setf8/setf16/rmif/cfinv/
    # axflag/xaflag) expose neither their NZCV reads nor their NZCV writes, and pacga omits its second
    # source (Xm). Fill them in precisely — only the flags each op truly writes, and every flag it
    # reads (see _FLAG_OP_NZCV / _rmif_written_flags): over-claiming a write drops live taint, a missed
    # read under-taints, both causing false violations.
    if mnemonic in _FLAG_OP_NZCV:
        reads, writes = _FLAG_OP_NZCV[mnemonic]
        src |= reads
        dest |= writes
        src.update(insn.reg_name(op.reg) for op in insn.operands if op.type == ARM64_OP_REG)
    elif mnemonic == "rmif":
        dest |= _rmif_written_flags(insn)
        src.update(insn.reg_name(op.reg) for op in insn.operands if op.type == ARM64_OP_REG)
    elif mnemonic == "pacga":
        src.update(insn.reg_name(op.reg) for op in insn.operands
                   if op.type == ARM64_OP_REG and not (op.access & CS_AC_WRITE))
    elif mnemonic in _MTE_FIRST_REG_DEST:
        regs = [insn.reg_name(op.reg) for op in insn.operands if op.type == ARM64_OP_REG]
        if regs:
            dest.add(regs[0])        # first register operand is the destination
            src.update(regs[1:])     # the rest are sources (a memory base is handled above)
        if mnemonic == "ldg":
            src.add(regs[0])         # LDG is RMW: loads the tag into Xt, preserving its other bits
        if mnemonic == "subps":
            dest |= FLAG_BITS

    return sorted(src), sorted(dest)


def decode_tag_store(encoding: int, pc: int):
    """For an STG-family tag store, return (mnemonic, xt_reg, base_reg, disp) from capstone's
    structured operands; else None. STG writes the granule's allocation tag (not data memory), so the
    target granule address is base_reg + disp; xt_reg supplies the tag. base+disp is correct for the
    offset, pre-index (disp set) and post-index (disp 0, EA == base) forms."""
    insns = list(_CAPSTONE.disasm(encoding.to_bytes(4, byteorder="little"), pc))
    if not insns or insns[0].mnemonic.lower() not in ("stg", "stzg", "st2g", "stz2g"):
        return None
    insn = insns[0]
    mem = next((o for o in insn.operands if o.type == ARM64_OP_MEM), None)
    xt = next((o for o in insn.operands if o.type == ARM64_OP_REG), None)
    if mem is None or xt is None:
        return None
    return insn.mnemonic.lower(), insn.reg_name(xt.reg), insn.reg_name(mem.mem.base), mem.mem.disp


def is_conditional_branch(encoding: int) -> bool:
    """Return True if encoding is a conditional branch: B.cond, CBZ/CBNZ (32/64), TBZ/TBNZ.
    B.cond with cond AL (0xE) or NV (0xF) branches unconditionally, so it is excluded."""
    op = (encoding >> 24) & 0xFF
    if op == 0x54:
        return (encoding & 0xF) < 0xE
    return op in (0x34, 0x35, 0xB4, 0xB5,        # CBZ/CBNZ w/x
                  0x36, 0x37, 0xB6, 0xB7)        # TBZ/TBNZ w/x


def disassemble_instruction(encoding: int, pc: int):
    try:
        code_bytes = encoding.to_bytes(4, byteorder="little")
        insns = list(_CAPSTONE.disasm(code_bytes, pc))
        if insns:
            insn = insns[0]
            return f"{insn.mnemonic} {insn.op_str}".strip()
        else:
            return "<unknown>"
    except Exception as e:
        return f"<decode error: {e}>"
