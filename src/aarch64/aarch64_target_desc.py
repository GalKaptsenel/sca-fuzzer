"""
File: aarch64-specific constants and lists

Copyright (C) Microsoft Corporation
SPDX-License-Identifier: MIT
"""
from typing import List
import re
import unicorn.arm64_const as ucc

from ..interfaces import Instruction, TargetDesc, MacroSpec, CPUDesc
from ..model import UnicornTargetDesc
from ..config import CONF

class Aarch64TargetDesc(TargetDesc):

    branch_conditions = {
    "EQ": ["", "", "", "r", "", "", "", "", ""],
    "NE": ["", "", "", "r", "", "", "", "", ""],
    "CS": ["r", "", "", "", "", "", "", "", ""],
    "CC": ["r", "", "", "", "", "", "", "", ""],
    "MI": ["", "", "", "", "r", "", "", "", ""],
    "PL": ["", "", "", "", "r", "", "", "", ""],
    "VS": ["", "", "", "", "", "", "", "", "r"],
    "VC": ["", "", "", "", "", "", "", "", "r"],
    "HI": ["r", "", "", "r", "", "", "", "", ""],
    "LS": ["r", "", "", "r", "", "", "", "", ""],
    "GE": ["", "", "", "", "r", "", "", "", "r"],
    "LT": ["", "", "", "", "r", "", "", "", "r"],
    "GT": ["", "", "", "r", "r", "", "", "", "r"],
    "LE": ["", "", "", "r", "r", "", "", "", "r"],
    "AL": ["", "", "", "", "", "", "", "", ""]
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
        "xzr": 64, "wzr": 32,
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
           for i in range(31)},
        "SP": {64: "sp", 32: "wsp"},
        "ZERO": {64: "xzr", 32: "wzr"},
        "PC": {64: "pc", 32: "pc", 16: "pc", 8: "pc"},
    }  # yapf: disable
    registers = {
        32:     [*[f"w{i}" for i in range(31)], "wzr"],
        64:     [*[f"x{i}" for i in range(31)], "xzr"],
    }  # yapf: disable

    simd_registers = {
        8:      [f"b{i}" for i in range(32)],
        16:     [f"h{i}" for i in range(32)],
        32:     [f"s{i}" for i in range(32)],
        64:     [f"d{i}" for i in range(32)],
        128:    [f"v{i}" for i in range(32)],
    }  # yapf: disable

    sve_scalable_vector_registers = [f"z{i}" for i in range(32)]
    sve_predicate_registers = [f"p{i}" for i in range(8)] # There are actually 16 of them, but some instruction only accept the lower half, for now I left it like this

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
        # "switch":
        #     MacroSpec(4, "switch", ("actor_id", "function_id", "", "")),
        # "set_k2u_target":
        #     MacroSpec(5, "set_k2u_target", ("actor_id", "function_id", "", "")),
        # "switch_k2u":
        #     MacroSpec(6, "switch_k2u", ("actor_id", "", "", "")),
        # "set_u2k_target":
        #     MacroSpec(7, "set_u2k_target", ("actor_id", "function_id", "", "")),
        # "switch_u2k":
        #     MacroSpec(8, "switch_u2k", ("actor_id", "", "", "")),
        # "set_h2g_target":
        #     MacroSpec(9, "set_h2g_target", ("actor_id", "function_id", "", "")),
        # "switch_h2g":
        #     MacroSpec(10, "switch_h2g", ("actor_id", "", "", "")),
        # "set_g2h_target":
        #     MacroSpec(11, "set_g2h_target", ("actor_id", "function_id", "", "")),
        # "switch_g2h":
        #     MacroSpec(12, "switch_g2h", ("actor_id", "", "", "")),
        # "landing_k2u":
        #     MacroSpec(13, "landing_k2u", ("", "", "", "")),
        # "landing_u2k":
        #     MacroSpec(14, "landing_u2k", ("", "", "", "")),
        # "landing_h2g":
        #     MacroSpec(15, "landing_h2g", ("", "", "", "")),
        # "landing_g2h":
        #     MacroSpec(16, "landing_g2h", ("", "", "", "")),
        # "set_data_permissions":
        #     MacroSpec(18, "set_data_permissions", ("actor_id", "int", "int", ""))
    }



  
    def __init__(self):
        super().__init__()
        # remove blocked registers
        filtered_decoding = {}
        for size, registers in self.registers.items():
            filtered_decoding[size] = []
            for register in registers:
                if register not in CONF.register_blocklist or register in CONF.register_allowlist:
                    filtered_decoding[size].append(register)
        self.registers = filtered_decoding

        vendor = "arm"
        model = "0xd46" # todo: manually added for now
        stepping = "1"

        self.cpu_desc = CPUDesc(vendor, model, family, stepping)

    @staticmethod
    def is_unconditional_branch(inst: Instruction) -> bool:
        return inst.name in ["B"]

    @staticmethod
    def is_call(inst: Instruction) -> bool:
        return inst.name in ["BL"]



    def __init__(self):
        super().__init__()
        # remove blocked registers
        filtered_decoding = {}
        for size, registers in self.registers.items():
            filtered_decoding[size] = []
            for register in registers:
                if register not in CONF.register_blocklist or register in CONF.register_allowlist:
                    filtered_decoding[size].append(register)
        self.registers = filtered_decoding

        # identify the CPU model we are running on
   #     with open("/proc/cpuinfo", "r") as f:
   #         cpuinfo = f.read()

   #         vendor_match = re.search(r"CPU implementer\s+:\s+(.*)", cpuinfo)
   #         assert vendor_match, "Failed to find vendor in /proc/cpuinfo"
   #         vendor = vendor_match.group(1)

   #         family_match = "not applicable"  # TODO: not applicable for aarch64
   #         assert family_match, "Failed to find family in /proc/cpuinfo"
        family = "0x1" #family_match.group(1)

   #         model_match = re.search(r"CPU part\s+:\s+(.*)", cpuinfo)
   #         assert model_match, "Failed to find model name in /proc/cpuinfo"
   #         model = model_match.group(1)

   #         stepping_match = re.search(r"CPU revision\s+:\s+(.*)", cpuinfo)
   #         assert stepping_match, "Failed to find stepping in /proc/cpuinfo"
   #         stepping = stepping_match.group(1)

        vendor = "arm"
        model = "0xd46" # todo: manually added for now
        stepping = "1"
        
        self.cpu_desc = CPUDesc(vendor, model, family, stepping)

    @staticmethod
    def is_unconditional_branch(inst: Instruction) -> bool:
        return inst.name in ["B"]

    @staticmethod
    def is_call(inst: Instruction) -> bool:
        return inst.name in ["BL"]


class Aarch64UnicornTargetDesc(UnicornTargetDesc):
    reg_str_to_constant = {
        "x0": ucc.UC_ARM64_REG_X0,
        "x1": ucc.UC_ARM64_REG_X1,
        "x2": ucc.UC_ARM64_REG_X2,
        "x3": ucc.UC_ARM64_REG_X3,
        "x4": ucc.UC_ARM64_REG_X4,
        "x5": ucc.UC_ARM64_REG_X5,
        "x6": ucc.UC_ARM64_REG_X6,
        "x7": ucc.UC_ARM64_REG_X7,
        "x8": ucc.UC_ARM64_REG_X8,
        "x9": ucc.UC_ARM64_REG_X9,
        "x10": ucc.UC_ARM64_REG_X10,
        "x11": ucc.UC_ARM64_REG_X11,
        "x12": ucc.UC_ARM64_REG_X12,
        "x13": ucc.UC_ARM64_REG_X13,
        "x14": ucc.UC_ARM64_REG_X14,
        "x15": ucc.UC_ARM64_REG_X15,
        "x16": ucc.UC_ARM64_REG_X16,
        "x17": ucc.UC_ARM64_REG_X17,
        "x18": ucc.UC_ARM64_REG_X18,
        "x19": ucc.UC_ARM64_REG_X19,
        "x20": ucc.UC_ARM64_REG_X20,
        "x21": ucc.UC_ARM64_REG_X21,
        "x22": ucc.UC_ARM64_REG_X22,
        "x23": ucc.UC_ARM64_REG_X23,
        "x24": ucc.UC_ARM64_REG_X24,
        "x25": ucc.UC_ARM64_REG_X25,
        "x26": ucc.UC_ARM64_REG_X26,
        "x27": ucc.UC_ARM64_REG_X27,
        "x28": ucc.UC_ARM64_REG_X28,
        "x29": ucc.UC_ARM64_REG_X29,
        "x30": ucc.UC_ARM64_REG_X30,
        "v0": ucc.UC_ARM64_REG_V0,
        "v1": ucc.UC_ARM64_REG_V1,
        "v2": ucc.UC_ARM64_REG_V2,
        "v3": ucc.UC_ARM64_REG_V3,
        "v4": ucc.UC_ARM64_REG_V4,
        "v5": ucc.UC_ARM64_REG_V5,
        "v6": ucc.UC_ARM64_REG_V6,
        "v7": ucc.UC_ARM64_REG_V7,
        "v8": ucc.UC_ARM64_REG_V8,
        "v9": ucc.UC_ARM64_REG_V9,
        "v10": ucc.UC_ARM64_REG_V10,
        "v11": ucc.UC_ARM64_REG_V11,
        "v12": ucc.UC_ARM64_REG_V12,
        "v13": ucc.UC_ARM64_REG_V13,
        "v14": ucc.UC_ARM64_REG_V14,
        "v15": ucc.UC_ARM64_REG_V15,
        "v16": ucc.UC_ARM64_REG_V16,
        "v17": ucc.UC_ARM64_REG_V17,
        "v18": ucc.UC_ARM64_REG_V18,
        "v19": ucc.UC_ARM64_REG_V19,
        "v20": ucc.UC_ARM64_REG_V20,
        "v21": ucc.UC_ARM64_REG_V21,
        "v22": ucc.UC_ARM64_REG_V22,
        "v23": ucc.UC_ARM64_REG_V23,
        "v24": ucc.UC_ARM64_REG_V24,
        "v25": ucc.UC_ARM64_REG_V25,
        "v26": ucc.UC_ARM64_REG_V26,
        "v27": ucc.UC_ARM64_REG_V27,
        "v28": ucc.UC_ARM64_REG_V28,
        "v29": ucc.UC_ARM64_REG_V29,
        "v30": ucc.UC_ARM64_REG_V30,
        "v31": ucc.UC_ARM64_REG_V31,

        "fp": ucc.UC_ARM64_REG_FP,
        "lr": ucc.UC_ARM64_REG_FP,

        "nzcv": ucc.UC_ARM64_REG_NZCV,
        "sp": ucc.UC_ARM64_REG_SP,
        "wsp": ucc.UC_ARM64_REG_WSP,
        "xzr": ucc.UC_ARM64_REG_XZR,
        "wzr": ucc.UC_ARM64_REG_WZR,
    }

    reg_decode = {
        "0": ucc.UC_ARM64_REG_X0,
        "1": ucc.UC_ARM64_REG_X1,
        "2": ucc.UC_ARM64_REG_X2,
        "3": ucc.UC_ARM64_REG_X3,
        "4": ucc.UC_ARM64_REG_X4,
        "5": ucc.UC_ARM64_REG_X5,
        "6": ucc.UC_ARM64_REG_X6,
        "7": ucc.UC_ARM64_REG_X7,
        "8": ucc.UC_ARM64_REG_X8,
        "9": ucc.UC_ARM64_REG_X9,
        "10": ucc.UC_ARM64_REG_X10,
        "11": ucc.UC_ARM64_REG_X11,
        "12": ucc.UC_ARM64_REG_X12,
        "13": ucc.UC_ARM64_REG_X13,
        "14": ucc.UC_ARM64_REG_X14,
        "15": ucc.UC_ARM64_REG_X15,
        "16": ucc.UC_ARM64_REG_X16,
        "17": ucc.UC_ARM64_REG_X17,
        "18": ucc.UC_ARM64_REG_X18,
        "19": ucc.UC_ARM64_REG_X19,
        "20": ucc.UC_ARM64_REG_X20,
        "21": ucc.UC_ARM64_REG_X21,
        "22": ucc.UC_ARM64_REG_X22,
        "23": ucc.UC_ARM64_REG_X23,
        "24": ucc.UC_ARM64_REG_X24,
        "25": ucc.UC_ARM64_REG_X25,
        "26": ucc.UC_ARM64_REG_X26,
        "27": ucc.UC_ARM64_REG_X27,
        "28": ucc.UC_ARM64_REG_X28,
        "29": ucc.UC_ARM64_REG_X29,
        "30": ucc.UC_ARM64_REG_X30,

        "V0": ucc.UC_ARM64_REG_V0,
        "V1": ucc.UC_ARM64_REG_V1,
        "V2": ucc.UC_ARM64_REG_V2,
        "V3": ucc.UC_ARM64_REG_V3,
        "V4": ucc.UC_ARM64_REG_V4,
        "V5": ucc.UC_ARM64_REG_V5,
        "V6": ucc.UC_ARM64_REG_V6,
        "V7": ucc.UC_ARM64_REG_V7,
        "V8": ucc.UC_ARM64_REG_V8,
        "V9": ucc.UC_ARM64_REG_V9,
        "V10": ucc.UC_ARM64_REG_V10,
        "V11": ucc.UC_ARM64_REG_V11,
        "V12": ucc.UC_ARM64_REG_V12,
        "V13": ucc.UC_ARM64_REG_V13,
        "V14": ucc.UC_ARM64_REG_V14,
        "V15": ucc.UC_ARM64_REG_V15,
        "V16": ucc.UC_ARM64_REG_V16,
        "V17": ucc.UC_ARM64_REG_V17,
        "V18": ucc.UC_ARM64_REG_V18,
        "V19": ucc.UC_ARM64_REG_V19,
        "V20": ucc.UC_ARM64_REG_V20,
        "V21": ucc.UC_ARM64_REG_V21,
        "V22": ucc.UC_ARM64_REG_V22,
        "V23": ucc.UC_ARM64_REG_V23,
        "V24": ucc.UC_ARM64_REG_V24,
        "V25": ucc.UC_ARM64_REG_V25,
        "V26": ucc.UC_ARM64_REG_V26,
        "V27": ucc.UC_ARM64_REG_V27,
        "V28": ucc.UC_ARM64_REG_V28,
        "V29": ucc.UC_ARM64_REG_V29,
        "V30": ucc.UC_ARM64_REG_V30,
        "V31": ucc.UC_ARM64_REG_V31,

        "FLAGS": ucc.UC_ARM64_REG_NZCV,
        "GEFLAGS": ucc.UC_ARM64_REG_NZCV,
        "NF": ucc.UC_ARM64_REG_NZCV,
        "ZF": ucc.UC_ARM64_REG_NZCV,
        "CF": ucc.UC_ARM64_REG_NZCV,
        "VF": ucc.UC_ARM64_REG_NZCV,
        "QF": ucc.UC_ARM64_REG_NZCV,
        "EFLAG": ucc.UC_ARM64_REG_NZCV,
        "AFLAG": ucc.UC_ARM64_REG_NZCV,
        "IRQFLAG": ucc.UC_ARM64_REG_NZCV,
        "FIQFLAG": ucc.UC_ARM64_REG_NZCV,
        "PEMODE": ucc.UC_ARM64_REG_NZCV,
        "SSBSFLAG": ucc.UC_ARM64_REG_NZCV,
        "PANFALG": ucc.UC_ARM64_REG_NZCV,
        "DITFLAG": ucc.UC_ARM64_REG_NZCV,

        "PC": ucc.UC_ARM64_REG_PC, # pseudo register
        "SP": ucc.UC_ARM64_REG_SP,
        "TPIDR_EL0": ucc.UC_ARM64_REG_TPIDR_EL0,
        "TPIDRRO_EL0": ucc.UC_ARM64_REG_TPIDRRO_EL0,
        "TPIDR_EL1": ucc.UC_ARM64_REG_TPIDR_EL1,
        "PSTATE": ucc.UC_ARM64_REG_PSTATE,
        "ELR_EL0": ucc.UC_ARM64_REG_ELR_EL0,
        "ELR_EL1": ucc.UC_ARM64_REG_ELR_EL1,
        "ELR_EL2": ucc.UC_ARM64_REG_ELR_EL2,
        "ELR_EL3": ucc.UC_ARM64_REG_ELR_EL3,
        "SP_EL0": ucc.UC_ARM64_REG_SP_EL0,
        "SP_EL1": ucc.UC_ARM64_REG_SP_EL1,
        "SP_EL2": ucc.UC_ARM64_REG_SP_EL2,
        "SP_EL3": ucc.UC_ARM64_REG_SP_EL3,
        "TTBR0_EL1": ucc.UC_ARM64_REG_TTBR0_EL1,
        "TTBR1_EL1": ucc.UC_ARM64_REG_TTBR1_EL1,
        "ESR_EL0": ucc.UC_ARM64_REG_ESR_EL0,
        "ESR_EL1": ucc.UC_ARM64_REG_ESR_EL1,
        "ESR_EL2": ucc.UC_ARM64_REG_ESR_EL2,
        "ESR_EL3": ucc.UC_ARM64_REG_ESR_EL3,
        "FAR_EL0": ucc.UC_ARM64_REG_FAR_EL0,
        "FAR_EL1": ucc.UC_ARM64_REG_FAR_EL1,
        "FAR_EL2": ucc.UC_ARM64_REG_FAR_EL2,
        "FAR_EL3": ucc.UC_ARM64_REG_FAR_EL3,
        "PAR_EL1": ucc.UC_ARM64_REG_PAR_EL1,
        "MAIR_EL1": ucc.UC_ARM64_REG_MAIR_EL1,
        "VBAR_EL0": ucc.UC_ARM64_REG_VBAR_EL0,
        "VBAR_EL1": ucc.UC_ARM64_REG_VBAR_EL1,
        "VBAR_EL2": ucc.UC_ARM64_REG_VBAR_EL2,
        "VBAR_EL3": ucc.UC_ARM64_REG_VBAR_EL3,

        "MSRS": -1,
    }

    registers: List[int] = [
        ucc.UC_ARM64_REG_X0, ucc.UC_ARM64_REG_X1, ucc.UC_ARM64_REG_X2, ucc.UC_ARM64_REG_X3,
        ucc.UC_ARM64_REG_X4, ucc.UC_ARM64_REG_X5, ucc.UC_ARM64_REG_NZCV, ucc.UC_ARM64_REG_SP
    ]
    simd128_registers: List[int] = [
        ucc.UC_ARM64_REG_V0, ucc.UC_ARM64_REG_V1, ucc.UC_ARM64_REG_V2, ucc.UC_ARM64_REG_V3,
        ucc.UC_ARM64_REG_V4, ucc.UC_ARM64_REG_V5, ucc.UC_ARM64_REG_V6, ucc.UC_ARM64_REG_V7,
        ucc.UC_ARM64_REG_V8, ucc.UC_ARM64_REG_V9, ucc.UC_ARM64_REG_V10, ucc.UC_ARM64_REG_V11,
        ucc.UC_ARM64_REG_V12, ucc.UC_ARM64_REG_V13, ucc.UC_ARM64_REG_V14, ucc.UC_ARM64_REG_V15,
        ucc.UC_ARM64_REG_V16, ucc.UC_ARM64_REG_V17, ucc.UC_ARM64_REG_V18, ucc.UC_ARM64_REG_V19,
        ucc.UC_ARM64_REG_V20, ucc.UC_ARM64_REG_V21, ucc.UC_ARM64_REG_V22, ucc.UC_ARM64_REG_V23,
        ucc.UC_ARM64_REG_V24, ucc.UC_ARM64_REG_V25, ucc.UC_ARM64_REG_V26, ucc.UC_ARM64_REG_V27,
        ucc.UC_ARM64_REG_V28, ucc.UC_ARM64_REG_V29, ucc.UC_ARM64_REG_V30, ucc.UC_ARM64_REG_V31
    ]
    barriers: List[str] = ['DMB', 'DSB', 'ISB', 'PSSBB', 'SB',
                           'LDAR', 'STLR', 'LDAXR', 'STLXR'] # One-way barrier
    flags_register: int = ucc.UC_ARM64_REG_NZCV
    pc_register: int = ucc.UC_ARM64_REG_PC
    sp_register: int = ucc.UC_ARM64_REG_SP
