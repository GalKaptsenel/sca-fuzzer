"""
File: AArch64 instruction disassembly helpers (capstone-based).
  - Decode a 32-bit encoding to text, read/written operands, branch class
"""
from typing import List, Tuple

from capstone import Cs, CS_ARCH_ARM64, CS_MODE_ARM, CS_AC_READ, CS_AC_WRITE
from capstone.arm64 import (ARM64_OP_REG, ARM64_OP_MEM,
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

    for op in insn.operands:
        if op.type == ARM64_OP_REG:
            reg = insn.reg_name(op.reg)
            if op.access & CS_AC_WRITE:
                dest.add(reg)
            if op.access & CS_AC_READ:
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

    # Capstone 5.0.x under-reports these: rmif/setf8/setf16 expose neither the source read nor the
    # NZCV write, and pacga omits its second source (Xm).
    mnemonic = insn.mnemonic.lower()
    if mnemonic in ("rmif", "setf8", "setf16"):
        dest |= FLAG_BITS
        src.update(insn.reg_name(op.reg) for op in insn.operands if op.type == ARM64_OP_REG)
    elif mnemonic == "pacga":
        src.update(insn.reg_name(op.reg) for op in insn.operands
                   if op.type == ARM64_OP_REG and not (op.access & CS_AC_WRITE))
    elif mnemonic in _MTE_FIRST_REG_DEST:
        regs = [insn.reg_name(op.reg) for op in insn.operands if op.type == ARM64_OP_REG]
        if regs:
            dest.add(regs[0])        # first register operand is the destination
            src.update(regs[1:])     # the rest are sources (a memory base is handled above)
        if mnemonic == "subps":
            dest |= FLAG_BITS

    return sorted(src), sorted(dest)


def is_conditional_branch(encoding: int) -> bool:
    """Return True if encoding is a conditional branch: B.cond, CBZ/CBNZ (32/64), TBZ/TBNZ."""
    op = (encoding >> 24) & 0xFF
    return op in (0x54,                          # B.cond
                  0x34, 0x35, 0xB4, 0xB5,        # CBZ/CBNZ w/x
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
