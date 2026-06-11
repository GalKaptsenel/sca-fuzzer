from __future__ import annotations
import re
from dataclasses import dataclass
from .models import MemAccess

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
    return AslSemantics(
        mem_access=_mem_access(asl),
        read_regvars=frozenset(reads),
        written_regvars=frozenset(writes),
        sp_regvars=frozenset(v.lower() for v in _SP_REGVAR.findall(asl)),
        flags_written=frozenset(flags_w),
        flags_read=frozenset(flags_r),
    )

