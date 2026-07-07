"""Download the ARM ISA and generate Revizor's base.json (AArch64).

The extractor (arm_isa_extractor) downloads ARM's machine-readable XML and extracts a general, ISA-only
IR; this module turns that IR into base.json: it computes each instruction's selection tags and writes
the operands in the shape isa_loader reads. Run via `Downloader(extensions, out_file).run()`.

Tags are the unit of instruction selection (a run requests a set; an instruction matches if it has ANY).
They are ISA-prefixed (BASE-/SVE-/SVE2-/SME-), coarse and fine across orthogonal axes:
  functional  : BASE-{ARITH,LOGICAL,SHIFT,BITFIELD,BITCOUNT,BITBYTE,CONDSEL,FLAGOP,CRC,PAC,MTE,MOVE,
                NOP,HINT,SYSTEM,BARRIER,EXCEPTION,FPSIMD} ; SVE-/SVE2-* ; SME
  memory      : coarse <ISA>-MEM + <ISA>-MEM-{LOAD,STORE} + kind <ISA>-MEM-{ATOMIC,EXCLUSIVE,ACQREL,COPY,SET}
  prefetch    : <ISA>-PREFETCH                          (a hint, kept outside the MEM family)
  control flow: BASE-BRANCH + one of BASE-BRANCH-{COND,CALL,RET,UNCOND} + BASE-BRANCH-INDIRECT if reg-target
  flags       : coarse <ISA>-FLAGS + <ISA>-FLAGS-{WRITE,READ}
Functional class is decided by name first (PAC prefix, flag-op/exception/barrier/hint sets) then by the
ARM `category` (whose "system" value is a grab-bag of PAC/hints/barriers/flag-ops/nop). Memory, prefetch,
branch and flag axes come from IR facts. Every instruction must classify, so tagging asserts a non-empty set.

base.json operand "values": REG -> register names; MEM -> base/index register names; IMM -> "bitmask" /
"[lo-hi:stride]" / explicit list; COND -> condition codes; FLAGS -> 9-slot NZCV r/w scheme.
"""
from __future__ import annotations
import json
import os
import tempfile
from ..interfaces import OT, OperandSpec, MemorySpec, InstructionSpec
from .aarch64_target_desc import AArch64MemRole
from .arm_isa_extractor import pipeline
from .arm_isa_extractor.models import OperandKind, MemRole, MemAccess

# ---------------------------------------------------------------------------
# Tagging
# ---------------------------------------------------------------------------

# functional classes within the 'general' category (a mnemonic may carry several, e.g. addg)
_GENERAL_CLASSES = {
    "BASE-ARITH": frozenset(
        "add adds sub subs adc adcs sbc sbcs madd msub maddpt msubpt addpt subpt smaddl smsubl umaddl"
        " umsubl smulh umulh sdiv udiv abs smax smin umax umin adr adrp".split()),
    "BASE-LOGICAL": frozenset("and ands orr orn eor eon bic bics".split()),
    "BASE-SHIFT": frozenset("lslv lsrv asrv rorv".split()),
    "BASE-BITFIELD": frozenset("bfm ubfm sbfm extr".split()),
    "BASE-BITCOUNT": frozenset("clz cls cnt ctz".split()),              # counters only; rev/rbit are BITBYTE
    "BASE-BITBYTE": frozenset("rbit rev rev16 rev32".split()),          # GP reversal; the NEON rev is FPSIMD
    "BASE-CONDSEL": frozenset("csel csinc csinv csneg ccmp ccmn".split()),
    "BASE-CRC": frozenset("crc32b crc32h crc32w crc32x crc32cb crc32ch crc32cw crc32cx".split()),
    "BASE-MOVE": frozenset("movz movk movn".split()),
}

# name-priority functional classes that span categories or have unique names (checked before category)
# MTE families (each instruction also carries the general "MTE" tag) — kept off the BASE-* tags so
# MTE generates only when an MTE category is enabled.
_MTE_ARITH   = frozenset("addg subg subp subps".split())                # tagged-pointer arithmetic
_MTE_TAG_MEM = frozenset("stg st2g stz2g stzg stgm stgp stzgm ldg ldgm".split())  # tag store + tag load
_MTE_BASE    = frozenset("irg gmi".split())                             # tag-register manipulation
_MTE = _MTE_ARITH | _MTE_TAG_MEM | _MTE_BASE
_FLAGOP = frozenset("rmif setf8 setf16 cfinv axflag xaflag".split())    # purpose is flag manipulation
_EXCEPTION = frozenset("brk hlt svc hvc smc dcps1 dcps2 dcps3 drps udf eret eretaa eretab".split())
_BARRIER = frozenset("dmb dsb isb sb tsb psb csdb dgh esb gcsb clrex ssbb pssbb".split())
_HINT = frozenset("hint yield wfe wfi wfet wfit sev sevl bti chkfeat clrbhb".split())  # NOP-space hints
# acquire/release ordering is not carried in the IR, so this mapping is name-based.
_ACQREL = frozenset(
    "ldar ldarb ldarh ldapr ldaprb ldaprh ldlar ldlarb ldlarh"
    " stlr stlrb stlrh stllr stllrb stllrh ldiapp stilp".split())
_INDIRECT_BRANCH = frozenset(
    "br braa braaz brab brabz blr blraa blraaz blrab blrabz ret retaa retab retaasppcr retabsppcr".split())

# SVE / SVE2 functional sub-classes (memory is derived structurally, not listed here)
_SVE_FUNC = {
    "SVE-ARITH": (
        "add sub mul sqadd uqadd sqsub uqsub abs neg sabd uabd smax smin umax umin fmax fmin fmaxnm"
        " fminnm fadd fsub fmul fdiv fabs fneg fmad fmla fmls fnmad fnmla fnmls fmulx frecps frsqrts"
        " sdot udot fdot mad mla mls smulh umulh sqmulh sqdmulh sqrdmulh asrd asr lsl lsr sqshl uqshl"
        " sqrshl uqrshl srshl urshl shadd uhadd srhadd urhadd shsub uhsub addhnb addhnt raddhnb raddhnt"
        " subhnb subhnt rsubhnb rsubhnt sadalp uadalp saddlb saddlt uaddlb uaddlt ssublb ssublt usublb"
        " usublt saddwb saddwt uaddwb uaddwt ssubwb ssubwt usubwb usubwt smullb smullt umullb umullt"
        " sqdmullb sqdmullt pmullb pmullt faddp fmaxp fminp fmaxnmp fminnmp addp frecpe frecpx frsqrte"
        " fcvt fcvtzs fcvtzu scvtf ucvtf fcvtlt fcvtnt fcvtx fcvtxnt bfcvt bfcvtnt bfdot bfmlalb bfmlalt"
        " bfmmla"),
    "SVE-LOGICAL": "and orr eor bic orn nor nand not cnot andv eorv orv eorbt eortb",
    "SVE-PREDICATE": (
        "ptrue pfalse pfirst whilelt whilele whilelo whilels whilege whilegt whilehi whilehs whilerw"
        " whilewr rdffr rdffrs setffr wrffr brka brkb brkn brkpa brkpb pnext psel"),
    "SVE-MOVE": "dup cpy sel mov insr ext splice compact",
    "SVE-REDUCE": "addv smaxv sminv umaxv uminv faddv fmaxv fminv fmaxnmv fminnmv andv eorv orv saddv uaddv",
    "SVE-BITCOUNT": "cls clz cnt cntp",
    "SVE-PERMUTE": ("trn1 trn2 zip1 zip2 uzp1 uzp2 rev tbl tbx lasta lastb clasta clastb"
                    " sunpkhi sunpklo uunpkhi uunpklo"),
    "SVE-COMPARE": ("cmpeq cmpne cmpgt cmpge cmplt cmple cmphi cmphs cmplo cmpls"
                    " fcmpeq fcmpne fcmpgt fcmpge fcmplt fcmple fcmuo"),
}
_SVE2_FUNC = {
    "SVE2-ARITH": ("srsra ursra ssra usra srsrh ursrh sqrshrn sqrshrnb sqrshrnt sqshrn sqshrnb sqshrnt"
                   " uqrshrn uqrshrnb uqrshrnt uqshrn uqshrnb uqshrnt sqrshr uqrshr sqdmlalb sqdmlalt"
                   " sqdmlslb sqdmlslt smlalb smlalt smlslb smlslt umlalb umlalt umlslb umlslt fmlalb"
                   " fmlalt fmlslb fmlslt sqdmullb sqdmullt"),
    "SVE2-CRYPTO": "aesd aese aesimc aesmc sha512h sha512h2 sha512su0 sha512su1 sm4e sm4ekey rax1 xar bcax eor3",
    "SVE2-BITMANIP": "bext bdep bgrp",
    "SVE2-HISTCNT": "histcnt histseg",
    "SVE2-MATCH": "match nmatch",
    "SVE2-MMLA": "smmla ummla usmmla",
}


def _invert(table: dict) -> dict:
    """mnemonic -> set(tags) from a {tag: iterable-of-mnemonics} table."""
    out: dict = {}
    for tag, names in table.items():
        for m in (names.split() if isinstance(names, str) else names):
            out.setdefault(m, set()).add(tag)
    return out


_GENERAL_MAP = _invert(_GENERAL_CLASSES)
_SVE_MAP = _invert(_SVE_FUNC)
_SVE2_MAP = _invert(_SVE2_FUNC)

_FPSIMD_CATS = frozenset({"advsimd", "float", "fpsimd"})
_SVE_CATS = frozenset({"sve", "sve2"})
_SME_CATS = frozenset({"mortlach", "mortlach2"})


def _functional(name: str, cat: str) -> set:
    """Functional/family class(es). Name-priority first (PAC prefix + flag-op/exception/barrier/hint),
    then the authoritative ARM category. Raises on an unknown category (loud, never a silent default)."""
    if name.startswith("xpac"):                     # strip a PAC (no others share these prefixes)
        return {"PAC", "PAC-STRIP"}
    if name.startswith("aut"):                       # authenticate
        return {"PAC", "PAC-AUTH"}
    if name.startswith("pac"):                       # sign (incl. pacga)
        return {"PAC", "PAC-SIGN"}
    if name in _MTE_ARITH:
        return {"MTE", "MTE-ARITH"}
    if name in _MTE_TAG_MEM:
        return {"MTE", "MTE-TAG-MEM"}
    if name in _MTE_BASE:
        return {"MTE", "MTE-BASE"}
    if name in _FLAGOP:
        return {"BASE-FLAGOP"}
    if name == "nop":
        return {"BASE-NOP"}
    if name in _EXCEPTION:
        return {"BASE-EXCEPTION"}
    if name in _BARRIER:
        return {"BASE-BARRIER"}
    if name in _HINT:
        return {"BASE-HINT"}
    if cat == "general":
        return set(_GENERAL_MAP.get(name, ()))      # may be empty -> a memory/flag/branch tag covers it
    if cat in _FPSIMD_CATS:
        return {"BASE-FPSIMD"}
    if cat == "system":
        return {"BASE-SYSTEM"}
    if cat in _SVE_CATS:
        return {"SVE"} | set(_SVE_MAP.get(name, ())) | set(_SVE2_MAP.get(name, ()))
    if cat in _SME_CATS:
        return {"SME"}
    raise KeyError(f"unknown ARM category {cat!r} for instruction {name!r}")


def get_tags(inst: dict) -> list:
    """Tags for one extractor-IR instruction dict (see module docstring for the taxonomy)."""
    name, cat = inst["name"], inst["category"]
    isa = "SVE" if cat in _SVE_CATS else "SME" if cat in _SME_CATS else "BASE"
    tags = _functional(name, cat)
    if name in _MTE:        # MTE is fully classified by family; no structural BASE-* tags
        return sorted(tags)

    # memory (load/store/rmw only; prefetch is separate)
    mem = inst["mem_access"]
    if mem in ("load", "store", "rmw", "ex-load", "ex-store"):
        tags.add(f"{isa}-MEM")
    if mem in ("load", "ex-load"):
        tags.add(f"{isa}-MEM-LOAD")                                # atomics are RMW, so not LOAD
    elif mem in ("store", "ex-store"):
        tags.add(f"{isa}-MEM-STORE")
    if mem in ("ex-load", "ex-store"):
        tags.add(f"{isa}-MEM-EXCLUSIVE")
    if mem == "rmw":                                               # rmw is always an LSE atomic or a MOPS copy
        tags.add(f"{isa}-MEM-COPY" if name.startswith("cpy") else f"{isa}-MEM-ATOMIC")
    elif mem == "store" and name.startswith("set"):                # MOPS set (setf* is mem=none, excluded)
        tags.add(f"{isa}-MEM-SET")
    if name in _ACQREL:
        tags.add(f"{isa}-MEM-ACQREL")

    if mem == "prefetch":                                          # a hint, not a memory access
        tags.add(f"{isa}-PREFETCH")

    # flags
    if inst["flags_written"] or inst["flags_read"]:
        tags.add(f"{isa}-FLAGS")
    if inst["flags_written"]:
        tags.add(f"{isa}-FLAGS-WRITE")
    if inst["flags_read"]:
        tags.add(f"{isa}-FLAGS-READ")

    # control flow: one mutually-exclusive kind, plus INDIRECT (register target) orthogonally
    if inst["control_flow"]:
        tags.add("BASE-BRANCH")
        if name.startswith(("b.", "bc.", "cb", "tb")):     # b.cc / bc.cc / compare-or-test-and-branch
            tags.add("BASE-BRANCH-COND")
        elif name.startswith("ret"):                       # return
            tags.add("BASE-BRANCH-RET")
        elif name.startswith("bl"):                        # bl/blr* — branch with link (call)
            tags.add("BASE-BRANCH-CALL")
        else:                                              # plain unconditional: b, br*
            tags.add("BASE-BRANCH-UNCOND")
        if name in _INDIRECT_BRANCH:
            tags.add("BASE-BRANCH-INDIRECT")

    if not tags:
        raise ValueError(f"instruction {name!r} (category {cat!r}) got no tags — classification gap")
    return sorted(tags)


# ---------------------------------------------------------------------------
# base.json materialisation
# ---------------------------------------------------------------------------

_FLAG_SLOTS = ("N", "Z", "C", "V")


def _reg_names(op: dict) -> list:
    """The asm-token prefix (`Xn`->x, `Qd`->q, `Zt`->z, ...) applied to each number in reg_range. A
    composite `<R><t>` token (no file letter) means both GP widths. The target descriptor later
    intersects this with the registers it allows, so out-of-pool numbers (e.g. 31=XZR/SP) drop out."""
    lo, hi, stride = op["reg_range"]
    prefix = op["name"][0].lower()
    if not prefix.isalpha():
        return [f"x{n}" for n in range(lo, hi + 1, stride)] + [f"w{n}" for n in range(lo, hi + 1, stride)]
    return [f"{prefix}{n}" for n in range(lo, hi + 1, stride)]


def _esizes(inst: dict) -> list:
    """Element sizes to fan this instruction's specs out over. Some immediates have a valid range that
    depends on the element width (SVE tsz-encoded shifts are valid in [0, esize-1]); the extractor tags
    each `imm_ranges` entry with its esize as `(esize, lo, hi, stride)`, where esize 0 means "applies to
    any element size" (the ordinary, single-range case). This returns the distinct non-zero esizes that
    any range-encoded immediate/extend operand varies over, or [0] when there are none. `_expand_instruction`
    emits one spec per returned esize, and `_imm_values(op, esize)` then selects the matching range — so a
    plain `ADD #imm` yields one spec while an SVE shift yields one per 8/16/32/64-bit element."""
    sizes = {r[0] for op in inst["operands"]
             if OperandKind(op["kind"]) in (OperandKind.IMM, OperandKind.EXTEND) and not op["values"]
             for r in op["imm_ranges"] if r[0]}
    return sorted(sizes) or [0]


# register files keyed by width-selector letter: GP (W/X, x0-x30) and scalar SIMD&FP (B/H/S/D/Q, b0-b31)
_REG_FILE = {"W": (32, "w", 31), "X": (64, "x", 31),
             "B": (8, "b", 32), "H": (16, "h", 32), "S": (32, "s", 32), "D": (64, "d", 32), "Q": (128, "q", 32)}
_SELECTORS = frozenset("R W X V".split())   # width selectors (R -> W/X ; V -> scalar SIMD B/H/S/D)


def _close(s: str, i: int) -> int:
    """Index of the bracket matching the open bracket `{` or `(` at s[i]."""
    closer = {"{": "}", "(": ")"}[s[i]]
    depth = 0
    for j in range(i, len(s)):
        if s[j] == s[i]:
            depth += 1
        elif s[j] == closer:
            depth -= 1
            if depth == 0:
                return j
    raise ValueError(f"unbalanced {s[i]!r} in {s!r}")


def _split_alternatives(s: str) -> list:
    """Split *s* on top-level '|' (ignoring any inside nested <>/{}/()/[] groups)."""
    parts, depth, cur = [], 0, ""
    for ch in s:
        depth += ch in "<{(["
        depth -= ch in ">})]"
        if ch == "|" and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    parts.append(cur)
    return parts


def _imm_values(op: dict, esize: int) -> list:
    """Every value of the operand's range for *esize*, expanded (stride included) so the generator picks
    from a concrete list. esize 0 (or no per-esize variation) uses the operand's sole range."""
    if op["values"]:
        return list(op["values"])
    ranges = op["imm_ranges"]
    if not ranges:
        raise ValueError(f"immediate {op['name']!r} has no value set")
    chosen = [r for r in ranges if r[0] == esize] or [r for r in ranges if r[0] == 0]
    if not chosen:                                    # no range for this element size -> invalid form
        raise ValueError(f"immediate {op['name']!r} has no range for element size {esize}")
    vals = set()
    for _e, lo, hi, stride in chosen:
        vals.update(range(lo, hi + 1, stride))
    return [str(v) for v in sorted(vals)]


# extractor IR operand kind -> Revizor operand type. An extend/shift modifier materialises as an
# immediate; whether an operand addresses memory is carried separately, by its MemoryRole.
_KIND_TO_OT = {OperandKind.REG: OT.REG, OperandKind.IMM: OT.IMM, OperandKind.EXTEND: OT.IMM,
               OperandKind.COND: OT.COND, OperandKind.LABEL: OT.LABEL}

# arm-specific: read/write direction of the memory *location* for each access kind. A prefetch is a
# hint touching no architectural state (its base is still address-computed, hence still sandboxed).
_ACCESS_DIR = {MemAccess.LOAD: (True, False), MemAccess.EX_LOAD: (True, False),
               MemAccess.STORE: (False, True), MemAccess.EX_STORE: (False, True),
               MemAccess.RMW: (True, True), MemAccess.PREFETCH: (False, False)}

# extractor address role -> arm MemoryRole subtype; NONE ("not an address component") -> Python None
_MEM_ROLE = {r: (None if r is MemRole.NONE else AArch64MemRole(r.value)) for r in MemRole}


def _operand_values(op: dict, kind: OperandKind, esize: int) -> list:
    if kind is OperandKind.REG:
        return _reg_names(op)
    if kind in (OperandKind.IMM, OperandKind.EXTEND):
        return _imm_values(op, esize)
    if kind is OperandKind.COND:
        return list(op["values"])
    if kind is OperandKind.LABEL:
        return []
    raise ValueError(f"unhandled operand kind {kind!r}")


def _operand(op: dict, esize: int) -> OperandSpec:
    """One OperandSpec with its real type (OT) and its own read/write (writeback already folded into
    write). An address component additionally carries its MemoryRole; ordinary operands carry none."""
    kind = OperandKind(op["kind"])
    return OperandSpec(_KIND_TO_OT[kind], op["width"], op["signed"], op["read"], op["write"],
                       _operand_values(op, kind, esize), op["name"], _MEM_ROLE[MemRole(op["mem_role"])])


def _memory_operand(components: list, mem_access: MemAccess) -> MemorySpec:
    """One memory access (the operands of a single `[...]`) as a MemorySpec wrapping its address
    components. The MemorySpec's src/dest is the access direction; the components keep their own
    read/write, so pre/post-index writeback on the base survives."""
    src, dest = _ACCESS_DIR[mem_access]
    return MemorySpec(components[0].width, False, src, dest, components, components[0].name)


# A register-offset's index width is tied to its extend modifier (the ARM `option` field): UXTW/SXTW
# take a 32-bit Wm, LSL/UXTX/SXTX take a 64-bit Xm. LSL/UXTX have no zero-shift asm form, so they need
# an explicit amount. The asm template offers both widths via `(<Wm>|<Xm>)` and the full extend set, so
# the valid pairing is resolved per expanded form (once the index width is fixed).
_EXTEND_INDEX_WIDTH = {"uxtw": 32, "sxtw": 32, "uxtx": 64, "sxtx": 64, "lsl": 64}
_EXTEND_NEEDS_AMOUNT = frozenset({"lsl", "uxtx"})


def _resolve_register_offset(mem_op: MemorySpec) -> bool:
    """Restrict a memory operand's extend values to those valid for its (now fixed) index register
    width, returning False if the form is invalid and must be dropped: a plain register offset (no
    extend) needs a 64-bit index, and an extend needs an index of its implied width (and an amount for
    LSL/UXTX). Leaves non-GP indices (e.g. SVE vector gathers) untouched."""
    comp = {c.mem_role: c for c in mem_op.inner}
    index = comp.get(AArch64MemRole.INDEX)
    if index is None or index.width not in (32, 64):
        return True
    extend = comp.get(AArch64MemRole.EXTEND)
    if extend is None:
        return index.width == 64
    has_amount = AArch64MemRole.OFFSET in comp
    extend.values = [e for e in extend.values if _EXTEND_INDEX_WIDTH[e] == index.width
                     and (has_amount or e not in _EXTEND_NEEDS_AMOUNT)]
    return bool(extend.values)


# Data-processing extended-register forms (ADD/SUB/...): the extend modifier names the source register
# width — UXT{B,H,W}/SXT{B,H,W} extend a 32-bit Wm, UXTX/SXTX a 64-bit Xm, and LSL keeps the operation
# (destination) width. The asm offers the source width via `<R><m>` and the full extend set, so the
# valid pairing is resolved per expanded form. (Memory address extends are handled above.)
_TRUE_EXTENDS = frozenset({"uxtb", "uxth", "uxtw", "uxtx", "sxtb", "sxth", "sxtw", "sxtx"})
_EXTEND_SRC_WIDTH = {"uxtb": 32, "uxth": 32, "uxtw": 32, "sxtb": 32, "sxth": 32, "sxtw": 32,
                     "uxtx": 64, "sxtx": 64}


def _extend_operand(operands: list):
    """The data-processing extend-modifier operand (its values name true extends), or None. The IR
    operand kind (extend vs plain shift) is not kept on OperandSpec, so identify it by its values."""
    for op in operands:
        if op.type is OT.IMM and set(op.values) & _TRUE_EXTENDS:
            return op
    return None


def _resolve_extend(operands: list, inst: dict) -> bool:
    """Correlate a data-processing extended-register form with its source register width, returning
    False if the form is invalid: an extend must match the source width (LSL matches the destination
    width), and a sub-width source with the optional extend omitted is not a legal form."""
    if any(isinstance(op, MemorySpec) for op in operands):
        return True                                   # address extends: see _resolve_register_offset
    extend = _extend_operand(operands)
    regs = [op for op in operands if op.type is OT.REG]
    if extend is not None:
        index = operands.index(extend)
        source = next((op for op in reversed(operands[:index]) if op.type is OT.REG), None)
        if source is None:
            raise ValueError(f"extend operand {extend.name!r} has no source register")
        width, dest_width = source.width, regs[0].width
        has_amount = any(op.type is OT.IMM for op in operands[index + 1:])

        def matches(v):                               # LSL keeps the operation width; the rest name it
            if v in _EXTEND_NEEDS_AMOUNT and not has_amount:
                return False                          # LSL/UXTX have no zero-shift asm form
            return width == dest_width if v == "lsl" else _EXTEND_SRC_WIDTH[v] == width

        extend.values = [v for v in extend.values if matches(v)]
        return bool(extend.values)
    # no extend in this form: if the instruction can extend and a source is narrower than the
    # operation width, the mandatory extend has been dropped -> not a legal form
    if regs and any(OperandKind(o["kind"]) is OperandKind.EXTEND and set(o["values"]) & _TRUE_EXTENDS
                    for o in inst["operands"]):
        return not any(r.width < regs[0].width for r in regs[1:])
    return True


def _resolve_bit_test(operands: list, control_flow: bool) -> bool:
    """A bit-test branch (TBZ/TBNZ — the only control-flow form carrying both a register and an
    immediate) tests bit #imm of a register, so the bit position must be below the register width
    (0-31 for a 32-bit Wt, 0-63 for Xt). Clamp the immediate to the tested register's width."""
    if not control_flow:
        return True
    reg = next((op for op in operands if op.type is OT.REG), None)
    imm = next((op for op in operands if op.type is OT.IMM), None)
    if reg is None or imm is None:
        return True
    imm.values = [v for v in imm.values if int(v) < reg.width]
    return bool(imm.values)


def _flags_operand(written: list, read: list) -> OperandSpec:
    """The implicit NZCV operand as Revizor's 9-slot r/w scheme (4 NZCV slots + 5 reserved)."""
    scheme = []
    for f in _FLAG_SLOTS:
        rw = ("r" if f in read else "") + ("w" if f in written else "")
        scheme.append("r/w" if rw == "rw" else rw)
    scheme += [""] * 5
    return OperandSpec(OT.FLAGS, 0, False, bool(read), bool(written), scheme, "flags")


def _parse_template(s: str) -> list:
    """Parse an ARM asm template into a flat node list. Node kinds:
       ('lit', text) | ('op', var) | ('wsel', 'R'|'W'|'X') | ('opt', nodes)
       | ('alt', [nodes, ...]) | ('list', nodes)  (a literal {…} register list)
       | ('mem', nodes)  (a `[...]` memory access; its operands form one MemorySpec)."""
    nodes, lit, i = [], "", 0

    def flush():
        nonlocal lit
        if lit:
            nodes.append(("lit", lit))
            lit = ""

    while i < len(s):
        c = s[i]
        if c == "<":                                   # <var>  (or <R>/<W>/<X> width selector)
            flush()
            j = s.index(">", i)
            var = s[i + 1:j]
            nodes.append(("wsel", var) if var in _SELECTORS else ("op", var))
            i = j + 1
        elif c == "{":
            flush()
            j = _close(s, i)
            inner = s[i + 1:j].strip()
            if inner in _SELECTORS:                     # brace-form width selector
                nodes.append(("wsel", inner))
            elif inner.startswith("<") and ("." in inner or "-" in inner):
                nodes.append(("list", _parse_template(s[i + 1:j])))   # literal register list
            else:
                nodes.append(("opt", _parse_template(s[i + 1:j])))    # optional group
            i = j + 1
        elif c == "(":                                 # (a|b) alternative ((s+1) lives inside <…>)
            flush()
            j = _close(s, i)
            alts = _split_alternatives(s[i + 1:j])
            nodes.append(("alt", [_parse_template(a) for a in alts]) if len(alts) > 1
                         else ("lit", s[i:j + 1]))
            i = j + 1
        elif c == "[":                                 # [ ... ] group; memory-vs-index decided from roles
            flush()
            j = s.index("]", i)
            nodes.append(("mem", _parse_template(s[i + 1:j])))
            i = j + 1
        else:
            lit += c
            i += 1
    flush()
    return nodes


def _expand_format(template: str, by_name: dict, esize: int, mem_access: MemAccess) -> list:
    """Enumerate every concrete asm form of *template* as (revizor_template, [OperandSpec]) pairs, by
    walking the parsed AST: optionals branch present/absent, alternatives branch per choice, a GP width
    selector (R->W and X; W/X fixed) prefixes the next register variable into a synthesised GP operand,
    and a `[...]` group folds its operands into one MemorySpec."""

    def reg_operand(var, prefix):
        # A width selector (<R>/<V>) only ever applies to a data/SIMD register: ARM writes a memory
        # index as an explicit (<Wm>|<Xm>) alternative, so a composite is never an address operand.
        width, letter, count = _REG_FILE[prefix]
        base = by_name[var]
        names = [f"{letter}{n}" for n in range(count)]
        return prefix + var, OperandSpec(OT.REG, width, False, base["read"], base["write"], names,
                                         prefix + var)

    def enum(nodes, prefix):
        if not nodes:
            return [("", [])]
        (kind, *payload), rest = nodes[0], nodes[1:]
        if kind == "lit":
            return [(payload[0] + t, ops) for t, ops in enum(rest, prefix)]
        if kind == "wsel":
            sel = payload[0]
            prefixes = ("W", "X") if sel == "R" else ("B", "H", "S", "D") if sel == "V" else (sel,)
            return [r for p in prefixes for r in enum(rest, p)]
        if kind == "op":
            var = payload[0]
            if prefix:                                 # the register this width selector applied to
                name, op = reg_operand(var, prefix)
            elif var in by_name:
                name, op = var, _operand(by_name[var], esize)
            else:                                      # placeholder with no operand -> loud, never silent
                raise KeyError(f"template placeholder <{var}> has no matching operand")
            return [("{" + name + "}" + t, [op] + ops) for t, ops in enum(rest, None)]
        if kind == "mem":                              # [ ... ] : a memory access, OR a SIMD/SVE lane index
            results = []
            for it, io in enum(payload[0], None):
                # the extractor assigns an addressing role only to a real memory-bracket component, so a
                # bracket is a memory access iff any operand in it carries a role; a lane/tile index
                # (e.g. `Vd.S[3]`, `ZA[Wv, offs]`) carries none -> keep it literal.
                address = any(o.mem_role is not None for o in io)
                for t, ops in enum(rest, None):
                    inner = [_memory_operand(io, mem_access)] if address else io
                    results.append(("[" + it + "]" + t, inner + ops))
            return results
        if kind == "opt":
            return enum(payload[0] + rest, prefix) + enum(rest, prefix)
        if kind == "alt":
            return [r for choice in payload[0] for r in enum(choice + rest, prefix)]
        if kind == "list":                             # literal {…} register list: escape braces for str.format
            return [("{{" + it + "}}" + t, io + ops)
                    for it, io in enum(payload[0], prefix) for t, ops in enum(rest, None)]
        raise ValueError(f"unknown node kind {kind!r}")

    return enum(_parse_template(template), None)


def _instruction(inst: dict, template: str, operands: list) -> InstructionSpec:
    implicit = ([_flags_operand(inst["flags_written"], inst["flags_read"])]
                if (inst["flags_written"] or inst["flags_read"]) else [])
    spec = InstructionSpec(inst["name"], inst["category"], inst["control_flow"], template=template,
                           operands=operands, implicit_operands=implicit, tags=tuple(get_tags(inst)))
    spec.constraints = tuple(tuple(c) for c in inst.get("constraints", ()))
    return spec


def _expand_instruction(inst: dict) -> list:
    """One spec per (element size x concrete asm format); duplicates are dropped."""
    by_name = {}
    for op in inst["operands"]:
        # the template references operands by name; if one name maps to differing definitions (e.g. SME
        # `LDR ZA[<Wv>, <offs>], [<Xn|SP>, #<offs>]` reuses <offs> as both a tile index and a memory
        # offset), the name-keyed format-trick can't represent it -> loud-skip, never a wrong spec.
        if by_name.get(op["name"], op) != op:
            raise ValueError(f"operand {op['name']!r} reused with conflicting definitions "
                             f"(roles {by_name[op['name']]['mem_role']} vs {op['mem_role']})")
        by_name[op["name"]] = op
    mem_access = MemAccess(inst["mem_access"])
    specs, seen = [], set()
    for esize in _esizes(inst):
        for template, operands in _expand_format(inst["asm_template"], by_name, esize, mem_access):
            # restrict extends to the (now fixed) register widths; drop invalid pairings
            if any(isinstance(o, MemorySpec) and not _resolve_register_offset(o) for o in operands):
                continue
            if not _resolve_extend(operands, inst):
                continue
            if not _resolve_bit_test(operands, inst["control_flow"]):
                continue
            template = template.rstrip()
            key = (template, tuple(o.name for o in operands))
            if key in seen:
                continue
            seen.add(key)
            specs.append(_instruction(inst, template, operands))
    return specs


def _serialize_operand(op: OperandSpec) -> dict:
    """Flatten an operand spec to the base.json shape isa_loader reads (the one place dicts appear)."""
    if isinstance(op, MemorySpec):
        return {"type_": op.type.name, "width": op.width, "signed": op.signed, "src": op.src,
                "dest": op.dest, "name": op.name, "inner": [_serialize_operand(c) for c in op.inner]}
    out = {"type_": op.type.name, "width": op.width, "signed": op.signed, "src": op.src,
           "dest": op.dest, "values": list(op.values), "name": op.name}
    if op.mem_role is not None:
        out["mem_role"] = op.mem_role.value
    return out


def _serialize(spec: InstructionSpec) -> dict:
    return {"name": spec.name, "category": spec.category, "control_flow": spec.control_flow,
            "template": spec.template, "tags": list(spec.tags),
            "constraints": [list(c) for c in spec.constraints],
            "operands": [_serialize_operand(o) for o in spec.operands],
            "implicit_operands": [_serialize_operand(o) for o in spec.implicit_operands]}


def _generatable(inst: dict) -> bool:
    """Whether Revizor can generate this instruction. A PC-relative reference (adr/adrp, literal load,
    PC-relative PAC) takes a label operand, but only a branch's label has a target the generator can
    place; such non-branch label forms are not generatable, so they are not emitted."""
    if not inst["control_flow"] and any(OperandKind(o["kind"]) is OperandKind.LABEL
                                        for o in inst["operands"]):
        return False
    return True


# SSBB / PSSBB are encoding aliases of DSB #0 / #4, so ARM's XML never emits them as their own
# mnemonics and the extractor drops them. Add them explicitly (modeled like SB/CSDB: BASE-BARRIER,
# no operands) so the generator can emit them for the barrier-honoring contracts.
_SYNTHETIC_BARRIERS = ("SSBB", "PSSBB")


def _synthetic_barrier_specs() -> list:
    specs = []
    for mnem in _SYNTHETIC_BARRIERS:
        spec = InstructionSpec(mnem.lower(), "system", False, template=mnem,
                               operands=[], implicit_operands=[], tags=("BASE-BARRIER",))
        spec.constraints = ()
        specs.append(spec)
    return specs


def generate(ir_path: str, out_path: str) -> dict:
    """Turn the extractor IR JSON at *ir_path* into Revizor's base.json at *out_path*. Returns
    {encoding_name: error} for encodings not yet representable (a template placeholder with no operand,
    e.g. SME ZA tiles; or an operand name reused with conflicting roles) — reported per encoding (so a
    skipped encoding doesn't implicate other encodings of the same mnemonic), never a broken spec."""
    ir = json.load(open(ir_path))
    specs, failures = [], {}
    for inst in ir:
        if not _generatable(inst):
            continue
        try:
            specs.extend(_expand_instruction(inst))
        except (KeyError, ValueError) as e:
            failures[inst["encoding_name"]] = str(e)
    have = {s.name for s in specs}
    specs.extend(s for s in _synthetic_barrier_specs() if s.name not in have)
    json.dump([_serialize(s) for s in specs], open(out_path, "w"), indent=1, sort_keys=True)
    return failures


# ---------------------------------------------------------------------------
# Downloader (Revizor's spec-download entry point for AArch64)
# ---------------------------------------------------------------------------

class Downloader:
    """`Downloader(extensions, out_file).run()` — download ARM's XML, extract the IR, write base.json."""

    def __init__(self, extensions: list, out_file: str) -> None:
        self.extensions = extensions          # instr-class categories (kept for interface compatibility)
        self.out_file = out_file

    def run(self) -> None:
        fd, ir_path = tempfile.mkstemp(suffix=".ir.json")
        os.close(fd)
        try:
            pipeline.run(ir_path)                       # download XML + extract the IR
            failures = generate(ir_path, self.out_file)  # IR -> base.json (tags + runtime operands)
        finally:
            os.unlink(ir_path)
        if failures:
            print(f"[instruction_db_builder] {len(failures)} instructions skipped (template placeholder with no "
                  f"operand, e.g. SME ZA tiles): {', '.join(sorted(failures))}")
