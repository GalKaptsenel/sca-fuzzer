"""
Static register and condition-code tables for AArch64.

All "what are the valid values for this operand kind" decisions live here,
keeping the parser free of hard-coded lists.
"""
from __future__ import annotations

# General-purpose registers.
# SP / XZR / WSP / WZR are excluded by default; the parser adds them back
# where the XML documentation explicitly mentions them.
GP_REGISTERS: dict[int, list[str]] = {
    32: [f"w{i}" for i in range(31)],
    64: [f"x{i}" for i in range(31)],
}

# SIMD / FP scalar registers
SIMD_REGISTERS: dict[int, list[str]] = {
    8:   [f"b{i}" for i in range(32)],
    16:  [f"h{i}" for i in range(32)],
    32:  [f"s{i}" for i in range(32)],
    64:  [f"d{i}" for i in range(32)],
    128: [f"q{i}" for i in range(32)],
}

# SVE
SVE_VECTOR_REGISTERS: list[str] = [f"z{i}" for i in range(32)]
SVE_PREDICATE_REGISTERS: list[str] = [f"p{i}" for i in range(16)]

# Standard ARM condition codes
CONDITION_CODES: list[str] = [
    "eq", "ne", "cs", "cc", "mi", "pl",
    "vs", "vc", "hi", "ls", "ge", "lt",
    "gt", "le", "al", "nv",
]

# Combined lookup: register name → bit width (for postprocessing)
REGISTER_WIDTHS: dict[str, int] = {
    **{f"x{i}": 64 for i in range(31)},
    **{f"w{i}": 32 for i in range(31)},
    **{f"v{i}": 128 for i in range(32)},
    **{f"q{i}": 128 for i in range(32)},
    **{f"d{i}": 64 for i in range(32)},
    **{f"s{i}": 32 for i in range(32)},
    **{f"h{i}": 16 for i in range(32)},
    **{f"b{i}": 8 for i in range(32)},
    "sp": 64, "wsp": 32, "xzr": 64, "wzr": 32,
    "fpsr": 32, "fpcr": 32,
    **{f"sctlr_el{i}": 64 for i in range(1, 4)},
    "ttbr0_el1": 64, "ttbr1_el1": 64, "tcr_el1": 64,
    "mair_el1": 64, "amair_el1": 64,
    **{f"vbar_el{i}": 64 for i in range(1, 4)},
    "spsr_el1": 64, "elr_el1": 64,
    **{f"tpidr_el{i}": 64 for i in range(3)},
    "tpidrro_el0": 64,
    **{f"dbgbvr{i}": 64 for i in range(16)},
    **{f"dbgbcr{i}": 32 for i in range(16)},
    **{f"dbgwvr{i}": 64 for i in range(16)},
    **{f"dbgwcr{i}": 32 for i in range(16)},
    "daif": 32, "esr_el1": 32, "esr_el2": 32,
    "far_el1": 64, "far_el2": 64,
    "icc_ctlr_el1": 32, "icc_iar1_el1": 32,
    "cntvct_el0": 64, "cntkctl_el1": 64, "pmccntr_el0": 64,
    "sysregs": 64,
}
