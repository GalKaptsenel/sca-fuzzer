"""
File: AArch64 instruction disassembly helpers (capstone-based).
  - Decode a 32-bit encoding to text, read/written operands, branch class
"""
from typing import Dict, List, Optional, Tuple

from ..interfaces import OT
from capstone import Cs, CS_ARCH_ARM64, CS_MODE_ARM, CS_AC_READ, CS_AC_WRITE
from capstone.arm64 import (ARM64_OP_REG, ARM64_OP_MEM, ARM64_OP_IMM,
                            ARM64_CC_INVALID, ARM64_CC_EQ, ARM64_CC_NE,
                            ARM64_CC_HS, ARM64_CC_LO, ARM64_CC_MI, ARM64_CC_PL,
                            ARM64_CC_VS, ARM64_CC_VC, ARM64_CC_HI, ARM64_CC_LS,
                            ARM64_CC_GE, ARM64_CC_LT, ARM64_CC_GT, ARM64_CC_LE,
                            ARM64_CC_AL, ARM64_CC_NV)

_CAPSTONE = Cs(CS_ARCH_ARM64, CS_MODE_ARM)
_CAPSTONE.detail = True

# Register-operand read/write roles are NOT re-derived here for instructions the ISA describes: they
# are the authoritative per-operand src/dest already carried by every Instruction/InstructionSpec
# (loaded from base.json). decode_reg_accesses takes that as `reg_roles` (see build_register_role_map)
# and applies it verbatim, so an RMW register (e.g. CAS's Rs, read AND written), a pure destination
# (an LSE LD<op>'s Rt), or a plain source are each categorised from the spec, never guessed. Capstone
# supplies only the register NAMES/order, the memory base/index, and the condition code; its op.access
# is used solely for instructions absent from the map (e.g. sealer-injected AND/ADD/XPAC helpers,
# which it reports correctly), falling back to an over-approximate read when even that is empty (a
# taint-safe direction: a missed read is unsafe, an extra read is not).

# Capstone 5.0.x does not mark SUBPS as writing NZCV (it is an MTE pointer-subtract that sets flags);
# no static register role covers this, so its full-NZCV write is added explicitly.
_FULL_NZCV_WRITERS = frozenset({"subps"})

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


def build_register_role_map(instructions) -> Dict[Tuple[str, int], Tuple[Tuple[bool, bool], ...]]:
    """Authoritative register-operand roles, taken straight from the ISA objects (no guessing).

    `instructions` is any iterable of objects exposing `.name` and `.operands`, where each operand has
    `.type`, `.src`, `.dest` -- both a concrete `Instruction` and an `InstructionSpec` qualify, so the
    map can be built from the loaded InstructionSet or from a test case's own instructions. The result
    keys on (mnemonic, number-of-register-operands) so different-arity forms of one mnemonic never
    collide, and maps to the (is_src, is_dest) of each register operand in operand order. Two specs that
    share a key must agree on roles (an architectural fact), else it fails loud rather than pick one.
    """
    roles: Dict[Tuple[str, int], Tuple[Tuple[bool, bool], ...]] = {}
    for inst in instructions:
        regs = [op for op in inst.operands if op.type == OT.REG]
        if not regs:
            continue
        key = (inst.name.lower(), len(regs))
        value = tuple((bool(op.src), bool(op.dest)) for op in regs)
        existing = roles.get(key)
        if existing is not None and existing != value:
            raise ValueError(f"conflicting register roles for {key}: {existing} vs {value}")
        roles[key] = value
    return roles


def decode_reg_accesses(encoding: int, pc: int,
                        reg_roles: Optional[Dict[Tuple[str, int], Tuple[Tuple[bool, bool], ...]]] = None
                        ) -> Tuple[List[str], List[str]]:
    """Register/flag reads (src) and writes (dest) of one encoded instruction.

    Register-operand roles come from `reg_roles` (build_register_role_map) -- the authoritative ISA
    roles -- when the instruction is present there. For instructions absent from the map (sealer-
    injected helpers, or when no map is supplied), Capstone's op.access is used, and an operand Capstone
    leaves access-empty is over-approximated as a read (taint-safe). Flag (NZCV) reads/writes are always
    derived here, since their per-bit set depends on the condition code / mask immediate that no static
    operand role captures.
    """
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

    # Register operands: authoritative roles from the ISA map when known, else Capstone's op.access.
    reg_ops = [op for op in insn.operands if op.type == ARM64_OP_REG]
    roles = reg_roles.get((mnemonic, len(reg_ops))) if reg_roles is not None else None
    if roles is not None and len(roles) != len(reg_ops):
        raise ValueError(f"{mnemonic}: role map has {len(roles)} register roles but decoded "
                         f"{len(reg_ops)} from 0x{encoding:08x}")
    for i, op in enumerate(reg_ops):
        reg = insn.reg_name(op.reg)
        if roles is not None:
            is_src, is_dest = roles[i]          # spec roles: read, written, or both (an RMW register)
            if is_src:
                src.add(reg)
            if is_dest:
                dest.add(reg)
        elif op.access & (CS_AC_READ | CS_AC_WRITE):
            if op.access & CS_AC_READ:
                src.add(reg)
            if op.access & CS_AC_WRITE:
                dest.add(reg)
        else:
            src.add(reg)                        # Capstone empty, no spec role: over-approx read (safe)

    for op in insn.operands:
        if op.type == ARM64_OP_MEM:
            if op.mem.base != 0:
                base_reg = insn.reg_name(op.mem.base)
                src.add(base_reg)
                # pre/post-index writeback also UPDATES the base register; Capstone leaves it out of
                # the operand access flags, so add it explicitly.
                if getattr(insn, "writeback", False):
                    dest.add(base_reg)
            if op.mem.index != 0:
                src.add(insn.reg_name(op.mem.index))

    # Flags (NZCV): register roles never carry per-bit flag precision, so derive it here. The
    # FEAT_FlagM/FlagM2 ops are under-reported by Capstone (neither NZCV reads nor writes exposed);
    # RMIF's write set depends on its mask immediate; SUBPS's full-NZCV write is not flagged at all.
    # Over-claiming a flag write drops live taint (false violation); a missed flag read under-taints.
    if mnemonic in _FLAG_OP_NZCV:
        reads, writes = _FLAG_OP_NZCV[mnemonic]
        src |= reads
        dest |= writes
    elif mnemonic == "rmif":
        dest |= _rmif_written_flags(insn)
    if mnemonic in _FULL_NZCV_WRITERS:
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
