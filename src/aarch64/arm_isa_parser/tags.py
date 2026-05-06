"""
Instruction tagging for AArch64.

Tags serve two purposes:
  1. Filtering — which instructions Revizor includes in a fuzzing run
  2. Analysis — grouping results by instruction class

Taxonomy design
---------------
Tags follow the pattern  <ISA>-<CLASS>  mirroring the x86 convention in
Revizor's instruction_categories list.  Each instruction can carry multiple
tags (e.g. an atomic load is both ATOMICS and MEMORY-LOAD).

Tag groups
----------
BASE-ARITH          Integer arithmetic (ADD, SUB, MUL, DIV, …)
BASE-LOGICAL        Bitwise logic (AND, ORR, EOR, BIC, …)
BASE-SHIFT          Shift / rotate (LSL, LSR, ASR, ROR variants)
BASE-BITFIELD       Bitfield manipulation (UBFM, SBFM, BFM, EXTR, …)
BASE-BITCOUNT       Bit-counting / reverse (CLZ, CLS, RBIT, REV, CNT, CTZ)
BASE-CONDSEL        Conditional select / compare (CSEL, CSINC, CCMP, RMIF, …)
BASE-BRANCH         All branches (B, BL, CBZ, TBZ, …)
BASE-COND-BRANCH    Conditional branches (B.cond, BC.cond, CBZ, CBNZ, TBZ, TBNZ)
BASE-UNCOND-BRANCH  Unconditional branches (B, BL, BLR*, BR*, RET*)
BASE-RET            Return instructions (RET, RETAA, RETAB, ERETAA, ERETAB, …)
BASE-MEMORY-LOAD    Plain loads (LDR, LDRB, LDP, …)
BASE-MEMORY-STORE   Plain stores (STR, STRB, STP, …)
BASE-ATOMIC         Atomic RMW ops (LDADD, CAS, SWP, …)
BASE-EXCLUSIVE      Exclusive monitor ops (LDXR, STXR, LDAXR, STLXR, …)
BASE-ACQUIRE        Acquire / release variants (LDAR, STLR, …)
BASE-CRC            CRC32 instructions
BASE-MTE            Memory Tagging Extension (IRG, GMI, LDG, STG, …)
BASE-PAC            Pointer Authentication (PACIA, AUTIA, PACDA, …)
BASE-COPY           Memory copy / set (CPYE, SETE, …)
BASE-SYSTEM         System / misc (RET, UDF, PRFM, …)
BASE-FLAG           Flag manipulation (SETF8, SETF16, …)
SVE-ARITH           SVE integer / FP arithmetic
SVE-LOGICAL         SVE bitwise logical
SVE-MEMORY-LOAD     SVE loads (LD1B, LD1H, LDFF1, …)
SVE-MEMORY-STORE    SVE stores (ST1B, ST1H, …)
SVE-PREDICATE       SVE predicate-generating (WHILELT, PTRUE, …)
SVE-MOVE            SVE move / broadcast (DUP, CPY, SEL, …)
SVE-REDUCE          SVE reduction (ADDV, SMAXV, …)
SVE-BITCOUNT        SVE bit-count (CNT, CLZ, CLS on vectors)
SVE-PERMUTE         SVE permute / shuffle (TRN, ZIP, UZP, REV, …)
SVE-COMPARE         SVE compare (CMPEQ, CMPGT, …)
SVE-MISC            SVE instructions not in the above groups
SVE2-ARITH          SVE2-only arithmetic (SRSRA, URSRA, …)
SVE2-CRYPTO         SVE2 crypto (SHA512, SM4, …)
SVE2-BITMANIP       SVE2 bit manipulation (BEXT, BDEP, BGRP, …)
SVE2-HISTCNT        SVE2 histogram (HISTCNT, HISTSEG)
SVE2-MATCH          SVE2 string match (MATCH, NMATCH)
SVE2-MMLA           SVE2 matrix multiply (SMMLA, UMMLA, USMMLA)
"""
from __future__ import annotations

from .models import InstructionSpec
from . import registers as regs

# ---------------------------------------------------------------------------
# Static sets for SVE register detection
# ---------------------------------------------------------------------------

_SVE_VREGS: frozenset[str] = frozenset(regs.SVE_VECTOR_REGISTERS)    # z0-z31
_SVE_PREGS: frozenset[str] = frozenset(regs.SVE_PREDICATE_REGISTERS)  # p0-p15
_SVE_REGS: frozenset[str] = _SVE_VREGS | _SVE_PREGS


# ---------------------------------------------------------------------------
# BASE instruction exact-match table
# ---------------------------------------------------------------------------

_PREFIX_RULES: list[tuple[str, tuple[str, ...]]] = [
    # --- branches ---
    ("BASE-BRANCH", ("b", "b.", "bc.", "bl", "blr", "blraa", "blraaz",
                     "blrab", "blrabz", "br", "braa", "braaz", "brab",
                     "brabz", "cbnz", "cbz", "tbnz", "tbz", "ret",
                     "retaa", "retab", "retaasppcr", "retabsppcr",
                     "eretaa", "eretab")),

    # --- arithmetic ---
    ("BASE-ARITH", ("add", "adds", "sub", "subs", "adc", "adcs", "sbc",
                    "sbcs", "madd", "msub", "maddpt", "msubpt", "addpt",
                    "subpt", "smaddl", "smsubl", "umaddl", "umsubl",
                    "smulh", "umulh", "sdiv", "udiv", "abs",
                    "smax", "smin", "umax", "umin",
                    "subg", "addg", "subp", "subps")),

    # --- logical ---
    ("BASE-LOGICAL", ("and", "ands", "orr", "orn", "eor", "eon",
                      "bic", "bics")),

    # --- shift / rotate ---
    ("BASE-SHIFT", ("lslv", "lsrv", "asrv", "rorv")),

    # --- bitfield ---
    ("BASE-BITFIELD", ("bfm", "ubfm", "sbfm", "extr")),

    # --- bit-counting / byte reversal ---
    ("BASE-BITCOUNT", ("clz", "cls", "rbit", "rev", "rev16", "rev32",
                       "cnt", "ctz")),

    # --- conditional select / compare ---
    ("BASE-CONDSEL", ("csel", "csinc", "csinv", "csneg",
                      "ccmp", "ccmn", "rmif",
                      "setf8", "setf16")),

    # --- CRC ---
    ("BASE-CRC", ("crc32b", "crc32h", "crc32w", "crc32x",
                  "crc32cb", "crc32ch", "crc32cw", "crc32cx")),

    # --- PAC (Pointer Authentication) ---
    ("BASE-PAC", ("pacia", "pacib", "pacda", "pacdb",
                  "pacga", "paciza", "pacizb", "pacdza", "pacdzb",
                  "pacia171615", "pacib171615",
                  "paciasppc", "pacibsppc",
                  "pacnbiasppc", "pacnbibsppc",
                  "autia", "autib", "autda", "autdb",
                  "autiza", "autizb", "autdza", "autdzb",
                  "autia171615", "autib171615",
                  "autiasppcr", "autibsppcr")),

    # --- MTE (Memory Tagging Extension) ---
    ("BASE-MTE", ("irg", "gmi", "addg", "subg", "subp", "subps",
                  "ldg", "ldgm", "stg", "st2g", "stz2g", "stzg",
                  "stgm", "stgp", "stzgm", "gcsstr", "gcssttr")),

    # --- memory copy / set (MOPS) ---
    ("BASE-COPY", ("cpye", "cpyf", "cpym", "cpyp",
                   "sete", "setg", "setm", "setp")),

    # --- exclusive monitor ---
    ("BASE-EXCLUSIVE", ("ldxr", "ldxrb", "ldxrh", "ldxp",
                        "stxr", "stxrb", "stxrh", "stxp",
                        "ldaxr", "ldaxrb", "ldaxrh", "ldaxp",
                        "stlxr", "stlxrb", "stlxrh", "stlxp",
                        "ldatxr", "ldtxr", "sttxr", "stltxr")),

    # --- acquire / release (non-exclusive) ---
    ("BASE-ACQUIRE", ("ldar", "ldarb", "ldarh",
                      "ldapr", "ldaprb", "ldaprh",
                      "ldlar", "ldlarb", "ldlarh",
                      "stlr", "stlrb", "stlrh",
                      "stllr", "stllrb", "stllrh",
                      "ldiapp", "stilp")),

    # --- atomics ---
    ("BASE-ATOMIC", ("ldadd", "ldclr", "ldeor", "ldset",
                     "ldsmax", "ldsmin", "ldumax", "ldumin",
                     "ldtadd", "ldtclr", "ldtset",
                     "swp", "swpp", "swpt",
                     "cas", "casp",
                     "ld64b", "st64b", "st64bv", "st64bv0",
                     "rcwcas", "rcwcasp", "rcwclr", "rcwclrp",
                     "rcwset", "rcwsetp", "rcwswp", "rcwswpp",
                     "rcwscas", "rcwscasp", "rcwsclr", "rcwsclrp",
                     "rcwsset", "rcwssetp", "rcwsswp", "rcwsswpp")),

    # --- plain loads ---
    ("BASE-MEMORY-LOAD", ("ldr", "ldrb", "ldrh", "ldrsb", "ldrsh", "ldrsw",
                          "ldp", "ldpsw", "ldtp", "ldtxr")),

    # --- plain stores ---
    ("BASE-MEMORY-STORE", ("str", "strb", "strh", "stp", "sttp")),

    # --- prefetch ---
    ("BASE-PREFETCH", ("prfm", "rprfm")),

    # --- system / misc ---
    ("BASE-SYSTEM", ("udf", "nop")),

    # --- flag manipulation ---
    ("BASE-FLAG", ("rmif", "setf8", "setf16")),
]

# Flatten for O(1) lookup: mnemonic → [tags]
_EXACT_TAG_MAP: dict[str, list[str]] = {}
for _tag, _mnemonics in _PREFIX_RULES:
    for _m in _mnemonics:
        _EXACT_TAG_MAP.setdefault(_m, []).append(_tag)


# ---------------------------------------------------------------------------
# Prefix-based fallback for large BASE families
# ---------------------------------------------------------------------------

_BASE_PREFIX_FALLBACKS: list[tuple[str, str]] = [
    ("BASE-COPY",         "cpy"),
    ("BASE-COPY",         "set"),
    ("BASE-ATOMIC",       "ldadd"),
    ("BASE-ATOMIC",       "ldclr"),
    ("BASE-ATOMIC",       "ldeor"),
    ("BASE-ATOMIC",       "ldset"),
    ("BASE-ATOMIC",       "ldsmax"),
    ("BASE-ATOMIC",       "ldsmin"),
    ("BASE-ATOMIC",       "ldumax"),
    ("BASE-ATOMIC",       "ldumin"),
    ("BASE-ATOMIC",       "ldtadd"),
    ("BASE-ATOMIC",       "ldtclr"),
    ("BASE-ATOMIC",       "ldtset"),
    ("BASE-ATOMIC",       "swp"),
    ("BASE-ATOMIC",       "cas"),
    ("BASE-ATOMIC",       "rcw"),
    ("BASE-MTE",          "setg"),
    ("BASE-MEMORY-LOAD",  "ldr"),
    ("BASE-MEMORY-STORE", "str"),
    ("BASE-BRANCH",       "b"),
    ("BASE-BRANCH",       "bl"),
    ("BASE-BRANCH",       "ret"),
    ("BASE-PAC",          "pac"),
    ("BASE-PAC",          "xpac"),
    ("BASE-PAC",          "aut"),
]


# ---------------------------------------------------------------------------
# SVE exact-match table
# ---------------------------------------------------------------------------

_SVE_EXACT: dict[str, list[str]] = {}

def _sve(tag: str, *mnemonics: str) -> None:
    for m in mnemonics:
        _SVE_EXACT.setdefault(m, []).append(tag)


# SVE arithmetic
_sve("SVE-ARITH",
     "add", "sub", "mul", "sqadd", "uqadd", "sqsub", "uqsub",
     "abs", "neg", "sabd", "uabd",
     "smax", "smin", "umax", "umin", "fmax", "fmin",
     "fmaxnm", "fminnm",
     "fadd", "fsub", "fmul", "fdiv", "fabs", "fneg",
     "fmad", "fmla", "fmls", "fnmad", "fnmla", "fnmls",
     "fmulx", "frecps", "frsqrts",
     "sdot", "udot", "fdot",
     "mad", "mla", "mls",
     "smulh", "umulh",
     "sqmulh", "sqdmulh", "sqrdmulh",
     "umulh",
     "asrd", "asr", "lsl", "lsr",
     "asr_wide", "lsl_wide", "lsr_wide",
     "sqshl", "uqshl", "sqrshl", "uqrshl",
     "srshl", "urshl",
     "shadd", "uhadd", "srhadd", "urhadd",
     "shsub", "uhsub",
     "addhnb", "addhnt", "raddhnb", "raddhnt",
     "subhnb", "subhnt", "rsubhnb", "rsubhnt",
     "sadalp", "uadalp",
     "saddlb", "saddlt", "uaddlb", "uaddlt",
     "ssublb", "ssublt", "usublb", "usublt",
     "saddwb", "saddwt", "uaddwb", "uaddwt",
     "ssubwb", "ssubwt", "usubwb", "usubwt",
     "smullb", "smullt", "umullb", "umullt",
     "sqdmullb", "sqdmullt",
     "pmullb", "pmullt",
     "faddp", "fmaxp", "fminp", "fmaxnmp", "fminnmp",
     "addp",
     "frecpe", "frecpx", "frsqrte",
     "fcvt", "fcvtzs", "fcvtzu", "scvtf", "ucvtf",
     "fcvtlt", "fcvtnt", "fcvtx", "fcvtxnt",
     "bfcvt", "bfcvtnt", "bfdot", "bfmlalb", "bfmlalt", "bfmmla",
     )

# SVE logical
_sve("SVE-LOGICAL",
     "and", "orr", "eor", "bic", "orn", "nor", "nand",
     "not", "cnot",
     "andv", "eorv", "orv",
     "eorbt", "eortb",
     )

# SVE memory loads
_sve("SVE-MEMORY-LOAD",
     "ld1b", "ld1h", "ld1w", "ld1d", "ld1q",
     "ld1sb", "ld1sh", "ld1sw",
     "ld1rqb", "ld1rqh", "ld1rqw", "ld1rqd",
     "ld1rob", "ld1roh", "ld1row", "ld1rod",
     "ldff1b", "ldff1h", "ldff1w", "ldff1d",
     "ldff1sb", "ldff1sh", "ldff1sw",
     "ldnf1b", "ldnf1h", "ldnf1w", "ldnf1d",
     "ldnf1sb", "ldnf1sh", "ldnf1sw",
     "ldnt1b", "ldnt1h", "ldnt1w", "ldnt1d",
     "ldnt1sb", "ldnt1sh", "ldnt1sw",
     "ld2b", "ld2h", "ld2w", "ld2d",
     "ld3b", "ld3h", "ld3w", "ld3d",
     "ld4b", "ld4h", "ld4w", "ld4d",
     "ldr",   # SVE predicate/vector register load
     )

# SVE memory stores
_sve("SVE-MEMORY-STORE",
     "st1b", "st1h", "st1w", "st1d", "st1q",
     "stnt1b", "stnt1h", "stnt1w", "stnt1d",
     "st2b", "st2h", "st2w", "st2d",
     "st3b", "st3h", "st3w", "st3d",
     "st4b", "st4h", "st4w", "st4d",
     "str",   # SVE predicate/vector register store
     )

# SVE predicate generation
_sve("SVE-PREDICATE",
     "ptrue", "pfalse", "pfirst",
     "whilelt", "whilele", "whilelo", "whilels",
     "whilege", "whilegt", "whilehi", "whilehs",
     "whilerw", "whilewr",
     "rdffr", "rdffrs", "setffr", "wrffr",
     "brka", "brkb", "brkn", "brkpa", "brkpb",
     "pnext",
     "psel",
     )

# SVE move / broadcast / select
_sve("SVE-MOVE",
     "dup", "cpy", "sel", "mov",
     "insr", "ext",
     "splice",
     "compact",
     )

# SVE reduction
_sve("SVE-REDUCE",
     "addv", "smaxv", "sminv", "umaxv", "uminv",
     "faddv", "fmaxv", "fminv", "fmaxnmv", "fminnmv",
     "andv", "eorv", "orv",
     "saddv", "uaddv",
     )

# SVE bit-count (vector)
_sve("SVE-BITCOUNT",
     "cls", "clz", "cnt", "cntp",
     )

# SVE permute / shuffle
_sve("SVE-PERMUTE",
     "trn1", "trn2",
     "zip1", "zip2",
     "uzp1", "uzp2",
     "rev",
     "tbl", "tbx",
     "lasta", "lastb",
     "clasta", "clastb",
     "sunpkhi", "sunpklo", "uunpkhi", "uunpklo",
     )

# SVE compare (produces predicate)
_sve("SVE-COMPARE",
     "cmpeq", "cmpne",
     "cmpgt", "cmpge", "cmplt", "cmple",
     "cmphi", "cmphs", "cmplo", "cmpls",
     "fcmpeq", "fcmpne",
     "fcmpgt", "fcmpge", "fcmplt", "fcmple",
     "fcmuo",
     )

# ---------------------------------------------------------------------------
# SVE prefix fallbacks (catch ld1b_gather, st1h_scatter, etc.)
# ---------------------------------------------------------------------------

_SVE_PREFIX_FALLBACKS: list[tuple[str, str]] = [
    ("SVE-MEMORY-LOAD",  "ld1"),
    ("SVE-MEMORY-LOAD",  "ld2"),
    ("SVE-MEMORY-LOAD",  "ld3"),
    ("SVE-MEMORY-LOAD",  "ld4"),
    ("SVE-MEMORY-LOAD",  "ldff"),
    ("SVE-MEMORY-LOAD",  "ldnf"),
    ("SVE-MEMORY-LOAD",  "ldnt"),
    ("SVE-MEMORY-STORE", "st1"),
    ("SVE-MEMORY-STORE", "st2"),
    ("SVE-MEMORY-STORE", "st3"),
    ("SVE-MEMORY-STORE", "st4"),
    ("SVE-MEMORY-STORE", "stnt"),
    ("SVE-PREDICATE",    "while"),
    ("SVE-PREDICATE",    "brk"),
    ("SVE-COMPARE",      "cmp"),
    ("SVE-COMPARE",      "fcmp"),
]

# ---------------------------------------------------------------------------
# SVE2-only mnemonics
# SVE2 is a superset of SVE, so SVE2 instructions also get the "SVE" tag.
# ---------------------------------------------------------------------------

_SVE2_EXACT: dict[str, list[str]] = {}

def _sve2(tag: str, *mnemonics: str) -> None:
    for m in mnemonics:
        _SVE2_EXACT.setdefault(m, []).append(tag)
        _SVE2_EXACT.setdefault(m, [])  # ensure key exists


_sve2("SVE2-ARITH",
      "srsra", "ursra", "ssra", "usra",
      "srsrh", "ursrh",
      "sqrshrn", "sqrshrnb", "sqrshrnt",
      "sqshrn", "sqshrnb", "sqshrnt",
      "uqrshrn", "uqrshrnb", "uqrshrnt",
      "uqshrn", "uqshrnb", "uqshrnt",
      "sqrshr", "uqrshr",
      "sqdmlalb", "sqdmlalt", "sqdmlslb", "sqdmlslt",
      "smlalb", "smlalt", "smlslb", "smlslt",
      "umlalb", "umlalt", "umlslb", "umlslt",
      "fmlalb", "fmlalt", "fmlslb", "fmlslt",
      "sqdmullb", "sqdmullt",
      )

_sve2("SVE2-CRYPTO",
      "aesd", "aese", "aesimc", "aesmc",
      "sha512h", "sha512h2", "sha512su0", "sha512su1",
      "sm4e", "sm4ekey",
      "rax1", "xar",
      "bcax", "eor3",
      )

_sve2("SVE2-BITMANIP",
      "bext", "bdep", "bgrp",
      )

_sve2("SVE2-HISTCNT",
      "histcnt", "histseg",
      )

_sve2("SVE2-MATCH",
      "match", "nmatch",
      )

_sve2("SVE2-MMLA",
      "smmla", "ummla", "usmmla",
      )

_SVE2_PREFIX_FALLBACKS: list[tuple[str, str]] = [
    ("SVE2-ARITH",   "srsra"),
    ("SVE2-ARITH",   "ursra"),
    ("SVE2-ARITH",   "sqrshr"),
    ("SVE2-ARITH",   "uqrshr"),
    ("SVE2-CRYPTO",  "aes"),
    ("SVE2-CRYPTO",  "sha512"),
    ("SVE2-CRYPTO",  "sm4"),
]


# ---------------------------------------------------------------------------
# Branch sub-classification helpers
# ---------------------------------------------------------------------------

_COND_BRANCH_EXACT: frozenset[str] = frozenset({"cbz", "cbnz", "tbz", "tbnz"})
_COND_BRANCH_PREFIXES: tuple[str, ...] = ("b.", "bc.")

_RET_EXACT: frozenset[str] = frozenset({
    "ret", "retaa", "retab", "retaasppcr", "retabsppcr",
    "eretaa", "eretab",
})

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_tags(inst: InstructionSpec) -> list[str]:
    """
    Return a sorted, deduplicated list of tags for *inst*.

    Resolution order
    ----------------
    1. SVE2 exact match  → adds SVE2-* tag + SVE tag
    2. SVE exact match   → adds SVE-* tag
    3. SVE operand inference (z/p registers present) → adds SVE + SVE-MISC
    4. SVE prefix fallback
    5. SVE2 prefix fallback
    6. BASE exact match
    7. BASE prefix fallback
    8. BASE operand-type inference (MEM, LABEL, FLAGS)
    9. BASE-MISC fallback
    """
    name = inst.name.lower()
    tags: set[str] = set()

    # Collect all operand values once for register-based inference
    all_values: frozenset[str] = frozenset(
        v for op in inst.operands for v in op.values
    )
    has_sve_regs = bool(all_values & _SVE_REGS)

    # ------------------------------------------------------------------
    # SVE2 exact match
    # ------------------------------------------------------------------
    if has_sve_regs and name in _SVE2_EXACT:
        tags.update(_SVE2_EXACT[name])
        tags.add("SVE")   # SVE2 ⊃ SVE

    # ------------------------------------------------------------------
    # SVE exact match
    # ------------------------------------------------------------------
    if has_sve_regs and name in _SVE_EXACT:
        tags.update(_SVE_EXACT[name])
        tags.add("SVE")

    # ------------------------------------------------------------------
    # SVE operand inference — catches instructions we haven't named
    # explicitly but that clearly use SVE registers
    # ------------------------------------------------------------------
    if has_sve_regs and not (tags & {"SVE", "SVE2-ARITH", "SVE2-CRYPTO",
                                      "SVE2-BITMANIP", "SVE2-HISTCNT",
                                      "SVE2-MATCH", "SVE2-MMLA"}):
        tags.add("SVE")
        tags.add("SVE-MISC")

    # ------------------------------------------------------------------
    # SVE prefix fallbacks
    # ------------------------------------------------------------------
    if has_sve_regs or "SVE" in tags:
        for tag, prefix in _SVE_PREFIX_FALLBACKS:
            if name.startswith(prefix) and tag not in tags:
                tags.add(tag)
                tags.add("SVE")
        for tag, prefix in _SVE2_PREFIX_FALLBACKS:
            if name.startswith(prefix) and tag not in tags:
                tags.add(tag)
                tags.add("SVE")

    # ------------------------------------------------------------------
    # BASE exact match (only for non-SVE instructions)
    # ------------------------------------------------------------------
    if not has_sve_regs:
        if name in _EXACT_TAG_MAP:
            tags.update(_EXACT_TAG_MAP[name])

        # BASE prefix fallback
        if not tags:
            for tag, prefix in _BASE_PREFIX_FALLBACKS:
                if name.startswith(prefix):
                    tags.add(tag)

    # ------------------------------------------------------------------
    # Operand-type inference (supplements both SVE and BASE)
    # ------------------------------------------------------------------
    has_mem   = any(op.type_ == "MEM"   for op in inst.operands)
    has_label = any(op.type_ == "LABEL" for op in inst.operands)
    has_flags = any(op.type_ == "FLAGS" for op in inst.implicit_operands)

    if inst.control_flow or has_label:
        tags.add("BASE-BRANCH")

    if has_flags and not tags & {"BASE-CONDSEL", "BASE-FLAG", "BASE-ARITH"}:
        tags.add("BASE-FLAG")

    if has_mem and not has_sve_regs:
        for op in inst.operands:
            if op.type_ == "MEM":
                if op.src:
                    tags.add("BASE-MEMORY-LOAD")
                if op.dest:
                    tags.add("BASE-MEMORY-STORE")
                break # classify based on first operand of the memory operand - workaround!

    # ------------------------------------------------------------------
    # Branch sub-classification (runs after all BASE-BRANCH assignments)
    # ------------------------------------------------------------------
    if "BASE-BRANCH" in tags:
        is_cond = name in _COND_BRANCH_EXACT or any(name.startswith(p) for p in _COND_BRANCH_PREFIXES)
        tags.add("BASE-COND-BRANCH" if is_cond else "BASE-UNCOND-BRANCH")
        if name in _RET_EXACT:
            tags.add("BASE-RET")

    # ------------------------------------------------------------------
    # Final fallback
    # ------------------------------------------------------------------
    if not tags:
        tags.add("BASE-MISC")

    return sorted(tags)


def tag_instructions(instructions: list[InstructionSpec]) -> None:
    """Add tags in-place to every instruction in the list."""
    for inst in instructions:
        inst.tags = get_tags(inst)
