from __future__ import annotations
import re
from dataclasses import dataclass
from .models import MemAccess, MemWidth

# ASL register access `<File>{<w>}(<v>)` / `<File>[<v>]`, e.g. `X{64}(t)`, `V{}(n)`, `P{}(g)`.
# File is one of X/W (GP), V/Q/D/S/H/B (SIMD&FP), Z/P (SVE), or the Vpart/ZA* SIMD/SME accessors;
# <v> (t, n, d, ...) is the reg-number variable Decode fills from the encoding, not a literal number.
_ACC = r"(?:Vpart|ZAtile|ZAslice|X|W|V|Z|P|Q|D|S|H|B)"   # longer names first
_ACC_OPEN = re.compile(_ACC + r"(?:\{[^}]*\}\(|\[)")     # an accessor up to its opening `(` / `[`
_REG_HELPER_READ = re.compile(r"(?:ShiftReg|ExtendReg)(?:\{[^}]*\})?\(\s*(\w+)")  # ShiftReg{}(m,...) -> read
# e.g. `if n == 31 then ... SP{64}()` => reg-var n is SP when 31 (else XZR)
_SP_REGVAR = re.compile(r"if\s+(\w+)\s*==\s*31\s+then(?:(?!\bend\b).)*?SP\{", re.S)
# AccessDescriptor states the access: CreateAccDesc<kind>(MemOp_LOAD|STORE; "Ex"=exclusive
_ACCDESC_MEMOP = re.compile(r"CreateAccDesc(\w*?)\(\s*MemOp_(\w+)", re.I)
_ACCDESC_KIND = re.compile(r"CreateAccDesc(\w+)")
# AccessDescriptor kinds whose access is single-copy-atomic and so requires NATURAL alignment (to its
# own access width; an unaligned one takes an Alignment fault regardless of SCTLR.A): the LSE/FP/
# read-check-write atomics, load/store-exclusive, and the acquire/release-ordered accesses (incl.
# limited-ordering-region and the 64-byte atomic). Plain GPR/SIMD/SVE/SME/MOPS accesses tolerate an
# unaligned address. LDGSTG tag STORES are handled apart: they align to the MTE tag granule, not a
# data width.
_ALIGN_ACCDESC = frozenset({"AtomicOp", "FPAtomicOp", "RCW", "ExLDST",
                            "AcqRel", "LDAcqPC", "ASIMDAcqRel", "LOR", "LS64"})
_TAG_ACCDESC = "LDGSTG"      # LDG (load, no alignment) / STG-family tag store (granule-aligned)
_FLAG_GROUP_W = re.compile(r"PSTATE\.\[([NZCV, ]+)\]\s*=", re.I)     # `PSTATE.[N,Z,C,V] =`
_FLAG_ONE_W = re.compile(r"PSTATE\.([NZCV])\s*=", re.I)             # `PSTATE.C =`
_FLAG_READ = re.compile(r"PSTATE\.([NZCV])(?!\s*=)", re.I)          # `PSTATE.C` as rvalue
_NZCV = frozenset("NZCV")


@dataclass(frozen=True)
class AslSemantics:
    mem_access: MemAccess
    read_regvars: frozenset     # reg-number vars read (src), e.g. {"n","s"}
    written_regvars: frozenset  # reg-number vars written (dest), e.g. {"t"}
    sp_regvars: frozenset       # reg-number vars that mean SP at 31 (else XZR)
    flags_written: frozenset
    flags_read: frozenset
    # width of the data transferred to/from memory (see MemWidth), or None when the instruction
    # transfers no register data (a prefetch hint, a block copy/set, or not a memory access).
    mem_width: MemWidth | None = None
    # how the access must be aligned: "natural" (to its access width -- single-copy-atomic accesses),
    # "granule" (to the MTE tag granule -- STG-family tag stores), or None (no alignment requirement).
    mem_alignment: str | None = None
    # the memory AccessDescriptor kind (e.g. "AtomicOp", "FPAtomicOp", "RCW", "GPR", "MOPS"), or None
    # for a non-memory instruction. Lets the DB builder split the memory family structurally instead of
    # by mnemonic.
    mem_accdesc: str | None = None


# A memory data transfer in the ASL is a `Mem{<w>}(...)` / `MemAtomic{<w>}(...)` accessor whose brace
# <w> is the access width IN BITS: a literal (a fixed sub-word access, e.g. ldrb -> `Mem{8}`),
# `datasize`/`elsize` (one data-register width, e.g. `Mem{datasize}`), or `2*datasize`/`2*elsize` (a
# register pair, e.g. ldp -> `Mem{2*datasize}`). A few atomics leave the brace empty (`MemAtomic{}`) and
# carry the width on the value they read instead: `let data : bits(<w>) = MemAtomic{}`.
_MEM_ACCESSOR = re.compile(r"Mem\w*\{([^}]*)\}")            # the accessor; group 1 = width expr (maybe "")
_MEM_EMPTY_WIDTH = re.compile(r"bits\(([^)]+)\)\s*=\s*Mem")  # empty-brace accessor: width on the value
_MEM_WIDTH_PAIR = re.compile(r"^\s*2\s*\*\s*(?:datasize|elsize)\s*$")
_MEM_WIDTH_REG = re.compile(r"^\s*(?:datasize|elsize)\s*$")


def _mem_access_width(asl: str) -> MemWidth | None:
    """The memory data-transfer width from the ASL accessor (see MemWidth). None when there is no
    accessor, or its width is not a scalar our model resolves (an SVE vector-length / element-loop
    access) -- those transfer no single fixed-width register value."""
    m = _MEM_ACCESSOR.search(asl)
    if m is None:
        return None
    expr = m.group(1).strip()
    if not expr:                                    # empty brace -> width is on the value read
        v = _MEM_EMPTY_WIDTH.search(asl)
        if v is None:
            return None
        expr = v.group(1).strip()
    if expr.isdigit():
        return MemWidth(const_bits=int(expr))
    if _MEM_WIDTH_PAIR.match(expr):
        return MemWidth(reg_mult=2)
    if _MEM_WIDTH_REG.match(expr):
        return MemWidth(reg_mult=1)
    return None                                     # non-scalar width (SVE VL / element loop): not modeled


def _mem_access(asl: str) -> MemAccess:
    if "CreateAccDescAtomicOp" in asl or "MemAtomic" in asl:
        return MemAccess.RMW
    if "MemCpyBytes" in asl:        # memory copy (MOPS): reads source and writes destination
        return MemAccess.RMW
    if "MemSetBytes" in asl:        # memory set (MOPS): writes destination
        return MemAccess.STORE
    m = _ACCDESC_MEMOP.search(asl)
    if m is not None:
        kind, op = m.group(1).lower(), m.group(2).upper()
        if "ex" in kind:
            return MemAccess.EX_LOAD if op == "LOAD" else MemAccess.EX_STORE
        if op in ("LOAD", "STORE"):
            return MemAccess.LOAD if op == "LOAD" else MemAccess.STORE
    if "PrefetchOp" in asl or "Prefetch(" in asl:
        return MemAccess.PREFETCH
    if re.search(r"Mem\w*\{[^}]*\}\([^)]*\)\s*=", asl):
        return MemAccess.STORE
    if re.search(r"=\s*Mem\w*\{", asl):
        return MemAccess.LOAD
    return MemAccess.NONE


_RENAME = re.compile(r"(?:let|var)\s+(\w+)\s*(?::[^=]*)?=\s*(\w+)\s*;")  # `let transfer = t;` pure rename


_INDEX_VAR = re.compile(r"[A-Za-z_][\w.]*")   # first identifier in an accessor index


def _reg_accesses(asl: str):
    """Yield (reg-var, is_write) for every register-file accessor in *asl*. For each `<File>{..}(` or
    `<File>[`, scan its index with balanced brackets to the matching close (so nested parens like
    `Z{VL}((t+r) MOD 32)` are handled in full), take the base reg-var (the index's first identifier,
    after the last dot so a struct field `memcpy.d` -> `d`), and call it a write iff a single `=`
    (not `==`) follows the close."""
    for m in _ACC_OPEN.finditer(asl):
        depth, j = 1, m.end()
        while j < len(asl) and depth:
            depth += (asl[j] in "([") - (asl[j] in ")]")
            j += 1
        word = _INDEX_VAR.search(asl[m.end():j - 1])  # the index text, between the brackets
        if word is None:
            continue                                  # no identifier (e.g. `SP{}()`, a literal) => not a reg-var
        var = word.group(0).rsplit(".", 1)[-1].lower()
        yield var, re.match(r"\s*=(?!=)", asl[j:]) is not None


def extract_asl_semantics(asl: str) -> AslSemantics:
    writes, reads = set(), set()
    for var, is_write in _reg_accesses(asl):
        (writes if is_write else reads).add(var)
    reads |= {i.lower() for i in _REG_HELPER_READ.findall(asl)}  # ShiftReg/ExtendReg are reads
    for alias, target in _RENAME.findall(asl):       # an access through a pure rename is an access to the var
        if alias.lower() in writes:
            writes.add(target.lower())
        if alias.lower() in reads:
            reads.add(target.lower())
    flags_w = set()
    for grp in _FLAG_GROUP_W.findall(asl):
        flags_w |= set(re.findall(r"[NZCV]", grp.upper()))
    flags_w |= {f.upper() for f in _FLAG_ONE_W.findall(asl)}
    flags_r = {f.upper() for f in _FLAG_READ.findall(asl)}
    if "ConditionHolds" in asl:
        flags_r |= _NZCV  # condition operand selects which; read footprint is all NZCV
    mem_access = _mem_access(asl)
    accdescs = _ACCDESC_KIND.findall(asl)
    mem_accdesc = accdescs[0] if accdescs else None
    if any(k in _ALIGN_ACCDESC for k in accdescs):
        mem_alignment = "natural"                       # single-copy-atomic: align to the access width
    elif _TAG_ACCDESC in accdescs and mem_access is MemAccess.STORE:
        mem_alignment = "granule"                       # STG-family tag store: align to the tag granule
    else:
        mem_alignment = None
    return AslSemantics(
        mem_access=mem_access,
        read_regvars=frozenset(reads),
        written_regvars=frozenset(writes),
        sp_regvars=frozenset(v.lower() for v in _SP_REGVAR.findall(asl)),
        flags_written=frozenset(flags_w),
        flags_read=frozenset(flags_r),
        mem_width=_mem_access_width(asl),
        mem_alignment=mem_alignment,
        mem_accdesc=mem_accdesc,
    )

