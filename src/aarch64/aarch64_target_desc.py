"""
File: x86-specific constants and lists

Copyright (C) Microsoft Corporation
SPDX-License-Identifier: MIT
"""
from typing import List
import re
import unicorn.x86_const as ucc  # type: ignore

from ..interfaces import Instruction, TargetDesc, MacroSpec, CPUDesc
from ..model import UnicornTargetDesc
from ..config import CONF

class X86TargetDesc(TargetDesc):
    register_sizes = {
        **{f"x{i}": 64 for i in range(31)},
        **{f"w{i}": 32 for i in range(31)},
        **{f"v{i}": 128 for i in range(32)},
        **{f"q{i}": 128 for i in range(32)},
        **{f"d{i}": 64 for i in range(32)},
        **{f"s{i}": 32 for i in range(32)},
        **{f"h{i}": 16 for i in range(32)},
        **{f"b{i}": 8 for i in range(32)},
        "sp": 64, "wsp": 32, "xzr": 64, "wzr": 32,
    }  # yapf: disable
    reg_normalized = {
        **{f"x{i}": f"X{i}" for i in range(31)},
        **{f"w{i}": f"X{i}" for i in range(31)},
        **{f"v{i}": f"V{i}" for i in range(32)},
        **{f"q{i}": f"V{i}" for i in range(32)},
        **{f"d{i}": f"V{i}" for i in range(32)},
        **{f"s{i}": f"V{i}" for i in range(32)},
        **{f"h{i}": f"V{i}" for i in range(32)},
        **{f"b{i}": f"V{i}" for i in range(32)},

        "nzcv": "FLAGS", "cpsr": "FLAGS", "fpsr": "FLAGS",
        "NF": "NF", "ZF": "ZF", "CF": "CF", "VF": "VF",
        "IF": "IF", "FF": "FF", "MF": "MF", "AF": "AF",
        "pc": "PC",
        "sp": "SP", "wsp": "SP",
        "xzr": "ZERO", "wzr": "ZERO",
        "fpcr": "FPCR",

        **{f"sctlr_el{i}": f"SCTLR_EL{i}" for i in range(1, 4)},
        "ttbr0_el1": "TTBR0_EL1", "ttbr1_el1": "TTBR1_EL1",
        "tcr_el1": "TCR_EL1",
        "mair_el1": "MAIR_EL1", "amair_el1": "AMAIR_EL1",
        **{f"vbar_el{i}": f"VBAR_EL{i}" for i in range(1, 4)},
        "spsr_el1": "SPSR_EL1", "elr_el1": "ELR_EL1",
        **{f"tpidr_el{i}": f"TPIDR_EL{i}" for i in range(3)},
        "tpidrro_el0": "TPIDRRO_EL0",
        **{f"dbgbvr{i}": f"DBGBVR{i}" for i in range(16)},
        **{f"dbgbcr{i}": f"DBGBCR{i}" for i in range(16)},
        **{f"dbgwvr{i}": f"DBGWVR{i}" for i in range(16)},
        **{f"dbgwcr{i}": f"DBGWCR{i}" for i in range(16)},
        "daif": "DAIF", "esr_el1": "ESR_EL1", "esr_el2": "ESR_EL2",
        "far_el1": "FAR_EL1", "far_el2": "FAR_EL2",
        "icc_ctlr_el1": "ICC_CTLR_EL1", "icc_iar1_el1": "ICC_IAR1_EL1",
        "cntvct_el0": "CNTVCT_EL0", "cntkctl_el1": "CNTKCTL_EL1", "pmccntr_el0": "PMCCNTR_EL0",
        "sysregs": "SYSREGS",

    }  # yapf: disable
    reg_denormalized = {
        **{f"X{i}": {64: f"x{i}", 32: f"w{i}"} for i in range(31)},
        **{f"V{i}": {128: f"v{i}", 64: f"d{i}", 32: f"s{i}", 16: f"h{i}", 8: f"b{i}"}
           for i in range(31)},
        "SP": {64: "sp", 32: "wsp"},
        "ZERO": {64: "xzr", 32: "wzr"},
        "PC": {64: "pc", 32: "pc", 16: "pc", 8: "pc"},
    }  # yapf: disable
    registers = {
        8:      [f"b{i}" for i in range(32)],
        16:     [f"h{i}" for i in range(32)],
        32:     [*[f"s{i}" for i in range(32)], *[f"w{i}" for i in range(31)]],
        64:     [*[f"d{i}" for i in range(32)], *[f"x{i}" for i in range(31)]],
        128:    [f"v{i}" for i in range(32)],
    }  # yapf: disable

    simd_registers = {
        8:      [f"b{i}" for i in range(32)],
        16:     [f"h{i}" for i in range(32)],
        32:     [f"s{i}" for i in range(32)],
        64:     [f"d{i}" for i in range(32)],
        128:    [f"v{i}" for i in range(32)],
    }  # yapf: disable

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
        "switch":
            MacroSpec(4, "switch", ("actor_id", "function_id", "", "")),
        "set_k2u_target":
            MacroSpec(5, "set_k2u_target", ("actor_id", "function_id", "", "")),
        "switch_k2u":
            MacroSpec(6, "switch_k2u", ("actor_id", "", "", "")),
        "set_u2k_target":
            MacroSpec(7, "set_u2k_target", ("actor_id", "function_id", "", "")),
        "switch_u2k":
            MacroSpec(8, "switch_u2k", ("actor_id", "", "", "")),
        "set_h2g_target":
            MacroSpec(9, "set_h2g_target", ("actor_id", "function_id", "", "")),
        "switch_h2g":
            MacroSpec(10, "switch_h2g", ("actor_id", "", "", "")),
        "set_g2h_target":
            MacroSpec(11, "set_g2h_target", ("actor_id", "function_id", "", "")),
        "switch_g2h":
            MacroSpec(12, "switch_g2h", ("actor_id", "", "", "")),
        "landing_k2u":
            MacroSpec(13, "landing_k2u", ("", "", "", "")),
        "landing_u2k":
            MacroSpec(14, "landing_u2k", ("", "", "", "")),
        "landing_h2g":
            MacroSpec(15, "landing_h2g", ("", "", "", "")),
        "landing_g2h":
            MacroSpec(16, "landing_g2h", ("", "", "", "")),
        "set_data_permissions":
            MacroSpec(18, "set_data_permissions", ("actor_id", "int", "int", ""))
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

        # identify the CPU model we are running on
        with open("/proc/cpuinfo", "r") as f:
            cpuinfo = f.read()
            if 'Intel' in cpuinfo:
                vendor = 'Intel'
            elif 'AMD' in cpuinfo:
                vendor = 'AMD'
            else:
                vendor = 'Unknown'

            family_match = re.search(r"cpu family\s+:\s+(.*)", cpuinfo)
            assert family_match, "Failed to find family in /proc/cpuinfo"
            family = family_match.group(1)

            model_match = re.search(r"model\s+:\s+(.*)", cpuinfo)
            assert model_match, "Failed to find model name in /proc/cpuinfo"
            model = model_match.group(1)

            stepping_match = re.search(r"stepping\s+:\s+(.*)", cpuinfo)
            assert stepping_match, "Failed to find stepping in /proc/cpuinfo"
            stepping = stepping_match.group(1)

        self.cpu_desc = CPUDesc(vendor, model, family, stepping)

        # select EPT/NPT bits based on the CPU vendor
        self.epte_bits = self.epte_bits_intel if vendor == 'Intel' else self.npte_bits_amd

    @staticmethod
    def is_unconditional_branch(inst: Instruction) -> bool:
        return inst.category == "BASE-UNCOND_BR"

    @staticmethod
    def is_call(inst: Instruction) -> bool:
        return inst.category == "BASE-CALL"


class X86UnicornTargetDesc(UnicornTargetDesc):
    reg_str_to_constant = {
        "al": ucc.UC_X86_REG_AL,
        "bl": ucc.UC_X86_REG_BL,
        "cl": ucc.UC_X86_REG_CL,
        "dl": ucc.UC_X86_REG_DL,
        "dil": ucc.UC_X86_REG_DIL,
        "sil": ucc.UC_X86_REG_SIL,
        "spl": ucc.UC_X86_REG_SPL,
        "bpl": ucc.UC_X86_REG_BPL,
        "ah": ucc.UC_X86_REG_AH,
        "bh": ucc.UC_X86_REG_BH,
        "ch": ucc.UC_X86_REG_CH,
        "dh": ucc.UC_X86_REG_DH,
        "ax": ucc.UC_X86_REG_AX,
        "bx": ucc.UC_X86_REG_BX,
        "cx": ucc.UC_X86_REG_CX,
        "dx": ucc.UC_X86_REG_DX,
        "di": ucc.UC_X86_REG_DI,
        "si": ucc.UC_X86_REG_SI,
        "sp": ucc.UC_X86_REG_SP,
        "bp": ucc.UC_X86_REG_BP,
        "eax": ucc.UC_X86_REG_EAX,
        "ebx": ucc.UC_X86_REG_EBX,
        "ecx": ucc.UC_X86_REG_ECX,
        "edx": ucc.UC_X86_REG_EDX,
        "edi": ucc.UC_X86_REG_EDI,
        "esi": ucc.UC_X86_REG_ESI,
        "esp": ucc.UC_X86_REG_ESP,
        "ebp": ucc.UC_X86_REG_EBP,
        "rax": ucc.UC_X86_REG_RAX,
        "rbx": ucc.UC_X86_REG_RBX,
        "rcx": ucc.UC_X86_REG_RCX,
        "rdx": ucc.UC_X86_REG_RDX,
        "rdi": ucc.UC_X86_REG_RDI,
        "rsi": ucc.UC_X86_REG_RSI,
        "rsp": ucc.UC_X86_REG_RSP,
        "rbp": ucc.UC_X86_REG_RBP,
        "xmm0": ucc.UC_X86_REG_XMM0,
        "xmm1": ucc.UC_X86_REG_XMM1,
        "xmm2": ucc.UC_X86_REG_XMM2,
        "xmm3": ucc.UC_X86_REG_XMM3,
        "xmm4": ucc.UC_X86_REG_XMM4,
        "xmm5": ucc.UC_X86_REG_XMM5,
        "xmm6": ucc.UC_X86_REG_XMM6,
        "xmm7": ucc.UC_X86_REG_XMM7,
        "xmm8": ucc.UC_X86_REG_XMM8,
        "xmm9": ucc.UC_X86_REG_XMM9,
        "xmm10": ucc.UC_X86_REG_XMM10,
        "xmm11": ucc.UC_X86_REG_XMM11,
        "xmm12": ucc.UC_X86_REG_XMM12,
        "xmm14": ucc.UC_X86_REG_XMM14,
        "xmm15": ucc.UC_X86_REG_XMM15,
    }

    reg_decode = {
        "A": ucc.UC_X86_REG_RAX,
        "B": ucc.UC_X86_REG_RBX,
        "C": ucc.UC_X86_REG_RCX,
        "D": ucc.UC_X86_REG_RDX,
        "DI": ucc.UC_X86_REG_RDI,
        "SI": ucc.UC_X86_REG_RSI,
        "SP": ucc.UC_X86_REG_RSP,
        "BP": ucc.UC_X86_REG_RBP,
        "8": ucc.UC_X86_REG_R8,
        "9": ucc.UC_X86_REG_R9,
        "10": ucc.UC_X86_REG_R10,
        "11": ucc.UC_X86_REG_R11,
        "12": ucc.UC_X86_REG_R12,
        "13": ucc.UC_X86_REG_R13,
        "14": ucc.UC_X86_REG_R14,
        "15": ucc.UC_X86_REG_R15,
        "FLAGS": ucc.UC_X86_REG_EFLAGS,
        "CF": ucc.UC_X86_REG_EFLAGS,
        "PF": ucc.UC_X86_REG_EFLAGS,
        "AF": ucc.UC_X86_REG_EFLAGS,
        "ZF": ucc.UC_X86_REG_EFLAGS,
        "SF": ucc.UC_X86_REG_EFLAGS,
        "TF": ucc.UC_X86_REG_EFLAGS,
        "IF": ucc.UC_X86_REG_EFLAGS,
        "DF": ucc.UC_X86_REG_EFLAGS,
        "OF": ucc.UC_X86_REG_EFLAGS,
        "AC": ucc.UC_X86_REG_EFLAGS,
        "XMM0": ucc.UC_X86_REG_XMM0,
        "XMM1": ucc.UC_X86_REG_XMM1,
        "XMM2": ucc.UC_X86_REG_XMM2,
        "XMM3": ucc.UC_X86_REG_XMM3,
        "XMM4": ucc.UC_X86_REG_XMM4,
        "XMM5": ucc.UC_X86_REG_XMM5,
        "XMM6": ucc.UC_X86_REG_XMM6,
        "XMM7": ucc.UC_X86_REG_XMM7,
        "XMM8": ucc.UC_X86_REG_XMM8,
        "XMM9": ucc.UC_X86_REG_XMM9,
        "XMM10": ucc.UC_X86_REG_XMM10,
        "XMM11": ucc.UC_X86_REG_XMM11,
        "XMM12": ucc.UC_X86_REG_XMM12,
        "XMM14": ucc.UC_X86_REG_XMM14,
        "XMM15": ucc.UC_X86_REG_XMM15,
        "RIP": -1,
        "RSP": -1,
        "CR0": -1,
        "CR2": -1,
        "CR3": -1,
        "CR4": -1,
        "CR8": -1,
        "XCR0": -1,
        "DR0": -1,
        "DR1": -1,
        "DR2": -1,
        "DR3": -1,
        "DR6": -1,
        "DR7": -1,
        "GDTR": -1,
        "IDTR": -1,
        "LDTR": -1,
        "TR": -1,
        "FSBASE": -1,
        "GSBASE": -1,
        "MSRS": -1,
        "X87CONTROL": -1,
        "TSC": -1,
        "TSCAUX": -1,
    }

    registers: List[int] = [
        ucc.UC_X86_REG_RAX, ucc.UC_X86_REG_RBX, ucc.UC_X86_REG_RCX, ucc.UC_X86_REG_RDX,
        ucc.UC_X86_REG_RSI, ucc.UC_X86_REG_RDI, ucc.UC_X86_REG_EFLAGS, ucc.UC_X86_REG_RSP
    ]
    simd128_registers: List[int] = [
        ucc.UC_X86_REG_XMM0, ucc.UC_X86_REG_XMM1, ucc.UC_X86_REG_XMM2, ucc.UC_X86_REG_XMM3,
        ucc.UC_X86_REG_XMM4, ucc.UC_X86_REG_XMM5, ucc.UC_X86_REG_XMM6, ucc.UC_X86_REG_XMM7,
        ucc.UC_X86_REG_XMM8, ucc.UC_X86_REG_XMM9, ucc.UC_X86_REG_XMM10, ucc.UC_X86_REG_XMM11,
        ucc.UC_X86_REG_XMM12, ucc.UC_X86_REG_XMM13, ucc.UC_X86_REG_XMM14, ucc.UC_X86_REG_XMM15
    ]
    barriers: List[str] = ['mfence', 'lfence']
    flags_register: int = ucc.UC_X86_REG_EFLAGS
    pc_register: int = ucc.UC_X86_REG_RIP
    actor_base_register: int = ucc.UC_X86_REG_R14
    sp_register: int = ucc.UC_X86_REG_RSP
