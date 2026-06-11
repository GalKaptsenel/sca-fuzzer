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
from .arm_isa_extractor import pipeline

# ---------------------------------------------------------------------------
# Tagging
# ---------------------------------------------------------------------------

# functional classes within the 'general' category (a mnemonic may carry several, e.g. addg)
_GENERAL_CLASSES = {
    "BASE-ARITH": frozenset(
        "add adds sub subs adc adcs sbc sbcs madd msub maddpt msubpt addpt subpt smaddl smsubl umaddl"
        " umsubl smulh umulh sdiv udiv abs smax smin umax umin subg addg subp subps adr adrp".split()),
    "BASE-LOGICAL": frozenset("and ands orr orn eor eon bic bics".split()),
    "BASE-SHIFT": frozenset("lslv lsrv asrv rorv".split()),
    "BASE-BITFIELD": frozenset("bfm ubfm sbfm extr".split()),
    "BASE-BITCOUNT": frozenset("clz cls cnt ctz".split()),              # counters only; rev/rbit are BITBYTE
    "BASE-BITBYTE": frozenset("rbit rev rev16 rev32".split()),          # GP reversal; the NEON rev is FPSIMD
    "BASE-CONDSEL": frozenset("csel csinc csinv csneg ccmp ccmn".split()),
    "BASE-CRC": frozenset("crc32b crc32h crc32w crc32x crc32cb crc32ch crc32cw crc32cx".split()),
    "BASE-MTE": frozenset("irg gmi addg subg subp subps ldg ldgm stg st2g stz2g stzg stgm stgp stzgm".split()),
    "BASE-MOVE": frozenset("movz movk movn".split()),
}

# name-priority functional classes that span categories or have unique names (checked before category)
_FLAGOP = frozenset("rmif setf8 setf16 cfinv axflag xaflag".split())    # purpose is flag manipulation
_EXCEPTION = frozenset("brk hlt svc hvc smc dcps1 dcps2 dcps3 drps udf eret eretaa eretab".split())
_BARRIER = frozenset("dmb dsb isb sb tsb psb csdb dgh esb gcsb clrex ssbb pssbb".split())
_HINT = frozenset("hint yield wfe wfi wfet wfit sev sevl bti chkfeat clrbhb".split())  # NOP-space hints
# acquire/release ordering is not yet carried in the IR, so this stays name-based for now.
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
    if name.startswith(("aut", "pac", "xpac")):     # every PAC/auth/strip variant (no others share these)
        return {"BASE-PAC"}
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

    assert tags, f"instruction {name!r} (category {cat!r}) got no tags — classification gap"
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
    """Distinct element sizes an immediate operand varies over (SVE); [0] when none (one spec)."""
    sizes = {r[0] for op in inst["operands"]
             if op["kind"] in ("imm", "extend") and not op["values"]
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
    chosen = [r for r in ranges if r[0] == esize] or [r for r in ranges if r[0] == 0] or ranges
    vals = set()
    for _e, lo, hi, stride in chosen:
        vals.update(range(lo, hi + 1, stride))
    return [str(v) for v in sorted(vals)]


def _operand(op: dict, esize: int) -> dict:
    kind = op["kind"]
    if kind == "reg":
        type_ = "MEM" if op["mem_role"] in ("base", "index") else "REG"   # address register stays MEM
        values = _reg_names(op)
    elif kind in ("imm", "extend"):
        type_, values = "IMM", _imm_values(op, esize)
    elif kind == "cond":
        type_, values = "COND", list(op["values"])
    elif kind == "label":
        type_, values = "LABEL", []
    else:
        raise ValueError(f"unhandled operand kind {kind!r}")
    return {"type_": type_, "width": op["width"], "signed": op["signed"],
            "src": op["read"], "dest": op["write"], "values": values, "name": op["name"]}


def _flags_operand(written: list, read: list) -> dict:
    """The implicit NZCV operand as Revizor's 9-slot r/w scheme (4 NZCV slots + 5 reserved)."""
    scheme = []
    for f in _FLAG_SLOTS:
        rw = ("r" if f in read else "") + ("w" if f in written else "")
        scheme.append("r/w" if rw == "rw" else rw)
    scheme += [""] * 5
    return {"type_": "FLAGS", "width": 0, "signed": False,
            "src": bool(read), "dest": bool(written), "values": scheme, "name": "flags"}


def _parse_template(s: str) -> list:
    """Parse an ARM asm template into a flat node list. Node kinds:
       ('lit', text) | ('op', var) | ('wsel', 'R'|'W'|'X') | ('opt', nodes)
       | ('alt', [nodes, ...]) | ('list', nodes)  (a literal {…} register list)."""
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
        else:
            lit += c
            i += 1
    flush()
    return nodes


def _expand_format(template: str, by_name: dict, esize: int) -> list:
    """Enumerate every concrete asm form of *template* as (revizor_template, [operand]) pairs, by walking
    the parsed AST: optionals branch present/absent, alternatives branch per choice, a GP width selector
    (R->W and X; W/X fixed) prefixes the next register variable into a synthesised GP operand."""

    def reg_operand(var, prefix):
        width, letter, count = _REG_FILE[prefix]
        base = by_name[var]
        return prefix + var, {"type_": "REG", "width": width, "signed": False,
                              "src": base["read"], "dest": base["write"],
                              "values": [f"{letter}{n}" for n in range(count)], "name": prefix + var}

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
        if kind == "opt":
            return enum(payload[0] + rest, prefix) + enum(rest, prefix)
        if kind == "alt":
            return [r for choice in payload[0] for r in enum(choice + rest, prefix)]
        if kind == "list":                             # literal {…} register list: escape braces for str.format
            return [("{{" + it + "}}" + t, io + ops)
                    for it, io in enum(payload[0], prefix) for t, ops in enum(rest, None)]
        raise ValueError(f"unknown node kind {kind!r}")

    return enum(_parse_template(template), None)


def _instruction(inst: dict, template: str, operands: list) -> dict:
    impl = ([_flags_operand(inst["flags_written"], inst["flags_read"])]
            if (inst["flags_written"] or inst["flags_read"]) else [])
    return {
        "name": inst["name"], "category": inst["category"], "control_flow": inst["control_flow"],
        "template": template, "operands": operands, "implicit_operands": impl,
        "tags": get_tags(inst),
        "constraints": [list(c) for c in inst.get("constraints", ())],
    }


def _expand_instruction(inst: dict) -> list:
    """One spec per (element size x concrete asm format); duplicates are dropped."""
    by_name = {op["name"]: op for op in inst["operands"]}
    specs, seen = [], set()
    for esize in _esizes(inst):
        for template, operands in _expand_format(inst["asm_template"], by_name, esize):
            template = template.rstrip()
            key = (template, tuple(o["name"] for o in operands))
            if key in seen:
                continue
            seen.add(key)
            specs.append(_instruction(inst, template, operands))
    return specs


def generate(ir_path: str, out_path: str) -> dict:
    """Turn the extractor IR JSON at *ir_path* into Revizor's base.json at *out_path*. Returns
    {name: error} for instructions whose template references a placeholder with no operand (a
    not-yet-modelled construct, e.g. SME ZA tiles / SVE multi-vector lists) — reported, never
    emitted as a broken spec."""
    ir = json.load(open(ir_path))
    specs, failures = [], {}
    for inst in ir:
        try:
            specs.extend(_expand_instruction(inst))
        except (KeyError, ValueError) as e:
            failures[inst["name"]] = str(e)
    json.dump(specs, open(out_path, "w"), indent=1, sort_keys=True)
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
            print(f"[isa_downloader] {len(failures)} instructions skipped (template placeholder with no "
                  f"operand, e.g. SME ZA tiles): {', '.join(sorted(failures))}")
