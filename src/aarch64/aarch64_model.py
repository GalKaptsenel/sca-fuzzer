"""
File: x86-specific model implementation

Copyright (C) Microsoft Corporation
SPDX-License-Identifier: MIT
"""
import re
import numpy as np
import copy
from typing import Tuple, Dict, List, Set, NamedTuple, Callable

import unicorn.arm64_const as ucc  # type: ignore
from unicorn import Uc, UcError, UC_MEM_WRITE, UC_ARCH_X86, UC_MODE_64, UC_PROT_READ, \
    UC_PROT_NONE, UC_ERR_WRITE_PROT, UC_ERR_NOMEM, UC_ERR_EXCEPTION, UC_ERR_INSN_INVALID

from ..interfaces import Input, FlagsOperand, RegisterOperand, MemoryOperand, AgenOperand, \
    TestCase, Instruction, Symbol, SANDBOX_DATA_SIZE, FAULTY_AREA_SIZE, OVERFLOW_PAD_SIZE, \
    UNDERFLOW_PAD_SIZE, SANDBOX_CODE_SIZE, get_sandbox_addr, ActorPL, InputTaint, CTrace, \
    ActorMode, UnreachableCode, NotSupportedException
from ..model import UnicornModel, UnicornTracer, UnicornSpec, UnicornSeq, BaseTaintTracker, \
    MacroInterpreter
from ..util import BLUE, COL_RESET, Logger, stable_hash_bytes
from ..config import CONF
from .aarch64_target_desc import Aarch64UnicornTargetDesc, Aarch64TargetDesc

# TODO: convert to aarch64 -> should I use cpsr or nzcv register?
FLAGS_CF = 0b000000000001
FLAGS_PF = 0b000000000100
FLAGS_AF = 0b000000010000
FLAGS_ZF = 0b000001000000
FLAGS_SF = 0b000010000000
FLAGS_TF = 0b000100000000
FLAGS_IF = 0b001000000000
FLAGS_DF = 0b010000000000
FLAGS_OF = 0b100000000000

CRITICAL_ERROR = UC_ERR_NOMEM  # the model never handles this error, hence it will always crash


class Aarch64MacroInterpreter(MacroInterpreter):
    pseudo_lstar: int
    curr_guest_target: int = 0
    curr_user_target: int = 0
    curr_host_target: int = 0

    def __init__(self, model: UnicornSeq):
        self.model = model
        self.is_arm = (model.target_desc.cpu_desc.vendor.lower() == "arm")

    def load_test_case(self, test_case: TestCase):
        self.test_case = test_case
        self.function_table = [symbol for symbol in test_case.symbol_table if symbol.type_ == 0]
        self.function_table.sort(key=lambda s: [s.arg])
        self.macro_table = [symbol for symbol in test_case.symbol_table if symbol.type_ != 0]
        self.sid_to_actor_name = {actor.id_: name for name, actor in test_case.actors.items()}
        self.pseudo_lstar = self.model.exit_addr

    def _get_macro_args(self, section_id: int, section_offset: int) -> Tuple[int, int, int, int]:
        # find the macro entry in the symbol table
        for symbol in self.macro_table:
            if symbol.aid == section_id and symbol.offset == section_offset:
                args = symbol.arg
                return args & 0xFFFF, (args >> 16) & 0xFFFF, (args >> 32) & 0xFFFF, \
                    (args >> 48) & 0xFFFF
        Logger().warning("get_macro_args", "macro not found in symbol table")
        raise UcError(CRITICAL_ERROR)

    def _find_function_by_id(self, function_id: int) -> Symbol:
        if function_id < 0 or function_id >= len(self.function_table):
            Logger().warning("find_function_by_id", "function not found in symbol table")
            raise UcError(CRITICAL_ERROR)
        return self.function_table[function_id]

    def interpret(self, macro: Instruction, address: int):
        macros: Dict[str, Callable] = {
            "measurement_start": self.macro_measurement_start,
            "measurement_end": self.macro_measurement_end,
            "switch": self.macro_switch,
            "switch_k2u": self.macro_switch_k2u,
            "switch_u2k": self.macro_switch_u2k,
            "set_k2u_target": self.macro_set_k2u_target,
            "set_u2k_target": self.macro_set_u2k_target,
            "switch_h2g": self.macro_switch_h2g,
            "switch_g2h": self.macro_switch_g2h,
            "set_h2g_target": self.macro_set_h2g_target,
            "set_g2h_target": self.macro_set_g2h_target,
            "landing_k2u": self.macro_landing_k2u,
            "landing_u2k": self.macro_landing_u2k,
            "landing_h2g": self.macro_landing_h2g,
            "landing_g2h": self.macro_landing_g2h,
            "fault_handler": lambda *_: None,
            "set_data_permissions": self.macro_set_data_permissions,
        }

        actor_id = self.model.current_actor.id_
        macro_offset = address - (self.model.code_start + SANDBOX_CODE_SIZE * actor_id)
        macro_args = self._get_macro_args(actor_id, macro_offset)

        interpreter_func = macros[macro.operands[0].value.lower()[1:]]
        interpreter_func(*macro_args)

    def macro_measurement_start(self, _: int, __: int, ___: int, ____: int):
        if not self.model.in_speculation:
            self.model.tracer.enable_tracing = True

    def macro_measurement_end(self, _: int, __: int, ___: int, ____: int):
        if not self.model.in_speculation:
            self.model.tracer.enable_tracing = False

    def macro_switch(self, section_id: int, function_id: int, _: int, __: int):
        """
        Switch the active actor, update data area base and SP,
          and jump to the corresponding function address
        """
        model = self.model
        section_addr = model.code_start + SANDBOX_CODE_SIZE * section_id

        # PC update
        function_symbol = self._find_function_by_id(function_id)
        function_addr = section_addr + function_symbol.offset
        model.emulator.reg_write(model.uc_target_desc.pc_register, function_addr)

        # data area base and SP update
        new_base = model.sandbox_base + SANDBOX_DATA_SIZE * section_id
        new_sp = get_sandbox_addr(new_base, "sp")
        model.emulator.reg_write(model.uc_target_desc.actor_base_register, new_base)
        model.emulator.reg_write(model.uc_target_desc.sp_register, new_sp)

        # actor update
        actor_name = self.sid_to_actor_name[section_id]
        model.current_actor = self.test_case.actors[actor_name]

    def macro_set_k2u_target(self, section_id: int, function_id: int, _: int, __: int):
        """
        Decode arguments and store destination into curr_user_target
        """
        section_addr = self.model.code_start + SANDBOX_CODE_SIZE * section_id
        function_symbol = self._find_function_by_id(function_id)
        function_addr = section_addr + function_symbol.offset
        self.curr_user_target = function_addr

    def macro_switch_k2u(self, section_id: int, _: int, __: int, ___: int):
        """ Read the destination from curr_user_target and jump to it;
        also update data area base and SP """
        model = self.model

        # PC update
        model.emulator.reg_write(model.uc_target_desc.pc_register, self.curr_user_target)

        # side effects
        # flags = model.emulator.reg_read(ucc.UC_X86_REG_EFLAGS)
        # rsp = model.emulator.reg_read(ucc.UC_X86_REG_RSP)
        # model.emulator.mem_write(rsp - 8, flags.to_bytes(8, byteorder='little'))  # type: ignore

        # data area base and SP update
        new_base = model.sandbox_base + SANDBOX_DATA_SIZE * section_id
        new_sp = get_sandbox_addr(new_base, "sp")
        model.emulator.reg_write(model.uc_target_desc.actor_base_register, new_base)
        model.emulator.reg_write(ucc.UC_X86_REG_RSP, new_sp)

        # actor update
        actor_name = self.sid_to_actor_name[section_id]
        model.current_actor = self.test_case.actors[actor_name]

    def macro_set_u2k_target(self, section_id: int, function_id: int, _: int, __: int):
        """ Set LSTAR to the target address if in kernel mode; otherwise, throw an exception """
        if self.model.current_actor.privilege_level != ActorPL.KERNEL:
            self.model.pending_fault_id = UC_ERR_EXCEPTION
            self.model.emulator.emu_stop()
            return
        model = self.model

        # update LSTAR
        section_addr = model.code_start + SANDBOX_CODE_SIZE * section_id
        function_symbol = self._find_function_by_id(function_id)
        function_addr = section_addr + function_symbol.offset
        self.pseudo_lstar = function_addr

    def macro_switch_u2k(self, section_id: int, _: int, __: int, ___: int):
        """ Switch the active actor, update data area base and SP, and jump to
            the pseudo_lstar
        """
        model = self.model

        # PC update
        model.emulator.reg_write(model.uc_target_desc.pc_register, self.pseudo_lstar)

        # data area base and SP update
        new_base = model.sandbox_base + SANDBOX_DATA_SIZE * section_id
        new_sp = get_sandbox_addr(new_base, "sp")
        model.emulator.reg_write(model.uc_target_desc.actor_base_register, new_base)
        model.emulator.reg_write(ucc.UC_X86_REG_RSP, new_sp)

        # actor update
        actor_name = self.sid_to_actor_name[section_id]
        model.current_actor = self.test_case.actors[actor_name]

    def macro_switch_h2g(self, section_id: int, _: int, __: int, ___: int):
        model = self.model

        # PC update
        model.emulator.reg_write(model.uc_target_desc.pc_register, self.curr_host_target)

        # data area base and SP update
        new_base = model.sandbox_base + SANDBOX_DATA_SIZE * section_id
        new_sp = get_sandbox_addr(new_base, "sp")
        model.emulator.reg_write(model.uc_target_desc.actor_base_register, new_base)
        model.emulator.reg_write(ucc.UC_X86_REG_RSP, new_sp)

        # reset flags
        model.emulator.reg_write(ucc.UC_X86_REG_EFLAGS, 0b10)

        # actor update
        actor_name = self.sid_to_actor_name[section_id]
        model.current_actor = self.test_case.actors[actor_name]

        # AMD VMRUN clobbers RAX; we model it as a zero write to RAX
        if self.is_amd:
            model.emulator.reg_write(ucc.UC_X86_REG_RAX, 0)

    def macro_switch_g2h(self, section_id: int, _: int, __: int, ___: int):
        model = self.model

        # PC update
        model.emulator.reg_write(model.uc_target_desc.pc_register, self.curr_guest_target)

        # data area base and SP update
        new_base = model.sandbox_base + SANDBOX_DATA_SIZE * section_id
        new_sp = get_sandbox_addr(new_base, "sp")
        model.emulator.reg_write(model.uc_target_desc.actor_base_register, new_base)
        model.emulator.reg_write(ucc.UC_X86_REG_RSP, new_sp)

        # actor update
        actor_name = self.sid_to_actor_name[section_id]
        model.current_actor = self.test_case.actors[actor_name]

        # AMD VMEXIT clobbers RAX; we model it as a zero write to RAX
        if self.is_amd:
            model.emulator.reg_write(ucc.UC_X86_REG_RAX, 0)

    def macro_set_h2g_target(self, section_id: int, function_id: int, _: int, __: int):
        section_addr = self.model.code_start + SANDBOX_CODE_SIZE * section_id
        function_symbol = self._find_function_by_id(function_id)
        function_addr = section_addr + function_symbol.offset
        self.curr_host_target = function_addr

    def macro_set_g2h_target(self, section_id: int, function_id: int, _: int, __: int):
        section_addr = self.model.code_start + SANDBOX_CODE_SIZE * section_id
        function_symbol = self._find_function_by_id(function_id)
        function_addr = section_addr + function_symbol.offset
        self.curr_guest_target = function_addr

    def macro_landing_k2u(self, _: int, __: int, ___: int, ____: int):
        """ Landing for the k2u switch """
        self.model.emulator.reg_write(ucc.UC_X86_REG_RCX, 0)

    def macro_landing_u2k(self, _: int, __: int, ___: int, ____: int):
        """ Landing for the u2k switch """
        self.model.emulator.reg_write(ucc.UC_X86_REG_RCX, 0)

    def macro_landing_h2g(self, _: int, __: int, ___: int, ____: int):
        """ Landing for the h2g switch """
        pass

    def macro_landing_g2h(self, _: int, __: int, ___: int, ____: int):
        """ Landing for the g2h switch """
        pass

    def macro_set_data_permissions(self, actor_id: int, must_set: int, must_clear: int, _: int):
        """ Manual setting of data permissions for the actor """
        pass


# ==================================================================================================
# Taint tracker
# ==================================================================================================
class X86TaintTracker(BaseTaintTracker):

    def __init__(self, initial_observations, sandbox_base=0):
        super().__init__(initial_observations, sandbox_base=sandbox_base)

        # ISA-specific field setup
        self.target_desc = X86TargetDesc()
        self.uc_target_desc = X86UnicornTargetDesc()

        self._registers = self.uc_target_desc.registers
        self._simd_registers = self.uc_target_desc.simd128_registers
