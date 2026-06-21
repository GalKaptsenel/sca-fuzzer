"""
File: aarch64-specific constants and lists
"""
from typing import List, Optional

from ..interfaces import Instruction, TargetDesc, MacroSpec, CPUDesc, MemoryRole
from ..config import CONF

# Parked in this shared constants module — a dedicated sandbox-ABI home would fit better.
SANDBOX_BASE_REGISTER = "x29"


class AArch64MemRole(MemoryRole):
    """AArch64 addressing roles inside `[...]` (the extractor's MemRole, minus NONE which maps to
    Python None)."""
    BASE = "base"      # address base register (the `<Xn|SP>`)
    INDEX = "index"    # address index register (the `<Xm>`)
    OFFSET = "offset"  # numeric address immediate (displacement or index shift amount)
    EXTEND = "extend"  # address index shift/extend type (LSL/ASR/UXTW/SXTW/...)


class Aarch64TargetDesc(TargetDesc):

    # Minimizer hooks. AArch64 uses GNU as default syntax (no header); the speculation fence is DSB SY.
    asm_header = ""
    speculation_barrier = "dsb sy"
    comment_symbol = "//"

    # Branch mnemonics whose neutralization would change control flow (kept out of NOP replacement).
    _BRANCH_MNEMONICS = frozenset(("b", "bl", "br", "blr", "ret", "cbz", "cbnz", "tbz", "tbnz"))

    def is_branch_line(self, line: str) -> bool:
        tokens = line.strip().lower().split()
        mnemonic = tokens[0] if tokens else ""
        return mnemonic in self._BRANCH_MNEMONICS or mnemonic.startswith("b.")

    def nop_replacement(self, line: str) -> Optional[str]:
        """Every AArch64 instruction is 4 bytes, so any non-branch instruction maps to a single NOP
        (offsets are preserved automatically)."""
        tokens = line.strip().lower().split()
        mnemonic = tokens[0] if tokens else ""
        if mnemonic == "nop" or self.is_branch_line(line):
            return None
        return "nop"

    branch_conditions = {
    "eq": ["", "", "", "r", "", "", "", "", ""],
    "ne": ["", "", "", "r", "", "", "", "", ""],
    "cs": ["r", "", "", "", "", "", "", "", ""],
    "cc": ["r", "", "", "", "", "", "", "", ""],
    "mi": ["", "", "", "", "r", "", "", "", ""],
    "pl": ["", "", "", "", "r", "", "", "", ""],
    "vs": ["", "", "", "", "", "", "", "", "r"],
    "vc": ["", "", "", "", "", "", "", "", "r"],
    "hi": ["r", "", "", "r", "", "", "", "", ""],
    "ls": ["r", "", "", "r", "", "", "", "", ""],
    "ge": ["", "", "", "", "r", "", "", "", "r"],
    "lt": ["", "", "", "", "r", "", "", "", "r"],
    "gt": ["", "", "", "r", "r", "", "", "", "r"],
    "le": ["", "", "", "r", "r", "", "", "", "r"],
    "al": ["", "", "", "", "", "", "", "", ""],
    "nv": ["", "", "", "", "", "", "", "", ""]
    }


    register_sizes = {
        **{f"x{i}": 64 for i in range(31)},
        **{f"w{i}": 32 for i in range(31)},
        **{f"v{i}": 128 for i in range(32)},
        **{f"q{i}": 128 for i in range(32)},
        **{f"d{i}": 64 for i in range(32)},
        **{f"s{i}": 32 for i in range(32)},
        **{f"h{i}": 16 for i in range(32)},
        **{f"b{i}": 8 for i in range(32)},
        **{f"SP_EL{i}": 64 for i in range(4)},
        **{f"ELR_EL{i}": 64 for i in range(1, 4)},
        "pc": 64,
        "xzr": 64, "wzr": 32, "sp": 64, "nzcv": 4
    }  # yapf: disable

    reg_normalized = {
        **{f"x{i}": f"{i}" for i in range(31)},
        **{f"w{i}": f"{i}" for i in range(31)},
        **{f"v{i}": f"V{i}" for i in range(32)},
        **{f"q{i}": f"V{i}" for i in range(32)},
        **{f"d{i}": f"V{i}" for i in range(32)},
        **{f"s{i}": f"V{i}" for i in range(32)},
        **{f"h{i}": f"V{i}" for i in range(32)},
        **{f"b{i}": f"V{i}" for i in range(32)},
        **{f"SP_EL{i}": "SP" for i in range(4)}, "sp": "SP",
        **{f"ELR_EL{i}": "ELR" for i in range(1, 4)},

        "nzcv": "FLAGS", "cpsr": "FLAGS", "fpsr": "FLAGS",
        "NF": "NF", "ZF": "ZF", "CF": "CF", "VF": "VF", "QF": "QF",
        "IRQFLAG": "IRQFLAG", "FIQFLAG": "FIQFLAG", "PEMODE": "PEMODE", "AFLAG": "AFLAG", "EFLAG": "EFLAG",
        "SSBSFLAG": "SSBSFLAG", "PANFLAG": "PANFLAG", "DITFLAG": "DITFLAG", "GEFLAGS": "GEFLAGS",
        "pc": "PC",
        "xzr": "ZERO", "wzr": "ZERO",
        "fpcr": "FPCR",

        **{f"dbgbvr{i}": f"DBGBVR{i}" for i in range(16)},
        **{f"dbgbcr{i}": f"DBGBCR{i}" for i in range(16)},
        **{f"dbgwvr{i}": f"DBGWVR{i}" for i in range(16)},
        **{f"dbgwcr{i}": f"DBGWCR{i}" for i in range(16)},

        **{f"sctlr_el{i}": f"SCTLR_EL{i}" for i in range(1, 4)},
        "ttbr0_el1": "TTBR0_EL1", "ttbr1_el1": "TTBR1_EL1",
        "tcr_el1": "TCR_EL1",
        "mair_el1": "MAIR_EL1", "amair_el1": "AMAIR_EL1",
        **{f"vbar_el{i}": f"VBAR_EL{i}" for i in range(1, 4)},
        "spsr_el1": "SPSR_EL1", "elr_el1": "ELR_EL1",
        **{f"tpidr_el{i}": f"TPIDR_EL{i}" for i in range(3)},
        "tpidrro_el0": "TPIDRRO_EL0",

        "daif": "DAIF", "esr_el1": "ESR_EL1", "esr_el2": "ESR_EL2",
        "far_el1": "FAR_EL1", "far_el2": "FAR_EL2",
        "icc_ctlr_el1": "ICC_CTLR_EL1", "icc_iar1_el1": "ICC_IAR1_EL1",
        "cntvct_el0": "CNTVCT_EL0", "cntkctl_el1": "CNTKCTL_EL1", "pmccntr_el0": "PMCCNTR_EL0",
        "sysregs": "SYSREGS",

    }  # yapf: disable
    reg_denormalized = {
        **{f"{i}": {64: f"x{i}", 32: f"w{i}"} for i in range(31)},
        **{f"V{i}": {128: f"v{i}", 64: f"d{i}", 32: f"s{i}", 16: f"h{i}", 8: f"b{i}"}
           for i in range(32)},
        "SP": {64: "sp", 32: "wsp"},
        "ZERO": {64: "xzr", 32: "wzr"},
        "PC": {64: "pc", 32: "pc", 16: "pc", 8: "pc"},
        "FLAGS": {4: "nzcv"},
    }  # yapf: disable
    # wzr/xzr are intentionally excluded: as a memory base/index they break the sandbox masking.
    registers = {
            32:     [*[f"w{i}" for i in range(31)]],
            64:     [*[f"x{i}" for i in range(31)]],
    }  # yapf: disable

    simd_registers = {
        8:      [f"b{i}" for i in range(32)],
        16:     [f"h{i}" for i in range(32)],
        32:     [f"s{i}" for i in range(32)],
        64:     [f"d{i}" for i in range(32)],
        128:    [f"v{i}" for i in range(32)],
    }  # yapf: disable

    sve_scalable_vector_registers = [f"z{i}" for i in range(32)]
    # Only p0–p7: many predicate operands accept just the lower half, and SVE is not currently
    # generated, so the lower 8 suffice (the architecture defines p0–p15).
    sve_predicate_registers = [f"p{i}" for i in range(8)]

    pte_bits = {
        # NAME: (position, default value)
        "present": (0, True),
        "is_table": (1, False),
        "attribute-index-in-mair-register-0": (2, False),
        "attribute-index-in-mair-register-1": (3, False),
        "attribute-index-in-mair-register-2": (4, False),
        "non-secure": (5, True),
        "unprivileged-has-access": (6, False),
        "privileged-read-only-access or dirty-bit": (7, False),
        "shareable": (8, False),
        "share_type": (9, False),
        "accessed": (10, False),
        "global": (11, False),
        "dirty-bit-modifier": (51, True),
        "Contiguous": (52, True),
        "privileged-execute-never": (53, True),
        "user-execute-never": (54, True),
    }

    # FIXME: macro IDs should not be hardcoded but rather received from the executor
    # or at least we need a test that will check that the IDs match
    macro_specs = {
        # macros with negative IDs are used for generation
        # and are not supposed to reach the final binary
        "random_instructions":
            MacroSpec(-1, "random_instructions", ("int", "int", "", "")),

        # macros with positive IDs are used for execution and can be interpreted by executor/model
        "function":
            MacroSpec(0, "function", ("", "", "", "")),
        "measurement_start":
            MacroSpec(1, "measurement_start", ("", "", "", "")),
        "measurement_end":
            MacroSpec(2, "measurement_end", ("", "", "", "")),
        "fault_handler":
            MacroSpec(3, "fault_handler", ("", "", "", "")),
    }

    def __init__(self):
        super().__init__()
        # remove blocked registers
        filtered_decoding = {}
        for size, regs in self.registers.items():
            filtered_decoding[size] = []
            for register in regs:
                if register not in CONF.register_blocklist or register in CONF.register_allowlist:
                    filtered_decoding[size].append(register)
        self.registers = filtered_decoding

        self.cpu_desc = self._read_cpu_desc()

    @staticmethod
    def _read_cpu_desc() -> CPUDesc:
        """Build the CPU descriptor from the MIDR fields Linux exposes per-core in /proc/cpuinfo.
        Only `vendor` is consulted by the aarch64 paths; model/family/stepping are descriptive.
        Loud-fail (no fallback) if the fields are absent — this is not an aarch64 host."""
        fields = {}
        with open("/proc/cpuinfo") as f:
            for line in f:
                key, sep, value = line.partition(":")
                if sep:
                    fields.setdefault(key.strip(), value.strip())  # first core; all cores are homogeneous
        try:
            return CPUDesc("aarch64", fields["CPU part"], fields["CPU variant"], fields["CPU revision"])
        except KeyError as missing:
            raise RuntimeError(f"/proc/cpuinfo missing MIDR field {missing} — not an aarch64 host?")

    @staticmethod
    def is_unconditional_branch(inst: Instruction) -> bool:
        return inst.name.lower() in ["b", "br", "ret"]

    @staticmethod
    def is_call(inst: Instruction) -> bool:
        return inst.name.lower() in ["bl", "blr"]
