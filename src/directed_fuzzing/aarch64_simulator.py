from __future__ import annotations
import warnings
import random

from typing import Optional, Set, Tuple, Union
from unicorn import (
    Uc, UC_ARCH_ARM64, UC_MODE_ARM, UC_ARM64_REG_PC,
    UC_ARM64_REG_SP, UC_ARM64_REG_NZCV
)
from keystone import Ks, KS_ARCH_ARM64, KS_MODE_LITTLE_ENDIAN

from .input_template import InputTemplate
from .value_strategy import ValueSelectionStrategy
from .arch_simulator import ArchSimStep, ArchSimulatorInterface, ArchSnapshotInterface
from ..interfaces import Instruction, Operand, OT


class UnicornSnapshot(ArchSnapshotInterface):
    """
    Stores touched registers (by reg_id) and 8-byte-aligned memory words.
    """
    def __init__(self, uas: UnicornArchSimulator, reg_ids: Set[int], addresses: Set[int]):
        self._uas = uas
        self._uc = uas._uc

        self._regs = {}      # reg_id -> int
        self._memory = {}    # aligned_addr -> bytes (8 bytes)

        
        if len(reg_ids) < 3: 
            warnings.warn(f"Check for a bug: SP, PC or NZCV not included in reg_ids: {reg_ids=}.")

        self._to_restore_regs = set(reg_ids) # Should copy
        self._to_restore_addresses = set(addresses) # Should copy

        for r in reg_ids:
            self._regs[r] = self._uc.reg_read(r)

        for addr in addresses:
            addr &= 7
            self._memory[addr] = self._uc.mem_read(addr, 8)

    def restore(self):
        self._uas._used_regs = self._to_restore_regs.copy()
        self._uas._used_addresses = self._to_restore_addresses.copy()

        for r, val in self._regs.items():
            self._uc.reg_write(r, val)

        for addr, data in self._memory.items():
            self._uc.mem_write(addr, data)

class AArch64Encoder:
    def __init__(self):
        self._ks = Ks(KS_ARCH_ARM64, KS_MODE_LITTLE_ENDIAN)

    def encode(self, asm: str, addr: int) -> bytes:
        encoding, _ = self._ks.asm(asm, addr=addr)
        return bytes(encoding)


class UnicornArchSimulator(ArchSimulatorInterface):
    """
    Architectural simulator with lazy value generation and minimal Unicorn resets.
    Assumptions about Operand:
      - op.type in OT.{REG,MEM}
      - if op.type == OT.REG: op.reg_id is unicorn reg id (int), op.value is register name (str)
      - if op.type == OT.MEM: op.offset is offset (int) relative to self._data_base(start of the sandbox memory) in bytes
      - op.is_src / op.is_dest booleans indicate source/destination roles
    Assumptions about InputTemplate:
      - template.is_concrete(key) accepts either str (reg name) or int (address)
      - template.set_concrete(key, value) accepts same keys
      - keys for memory are integer addresses (does not have to be aligned!)

    Notice:
        For snapshot we are aligning to 8 bytes, and snapshot the entire byte
    """

    def __init__(
        self,
        arch: int = UC_ARCH_ARM64,
        mode: int = UC_MODE_ARM,
        value_strategy: Optional[ValueSelectionStrategy] = None,
        rng_seed: Optional[int] = None,
        initial_pages: Optional[Tuple[int, int]] = None
    ):
        self._uc = Uc(arch, mode)
        self._value_strategy = value_strategy
        self._rnd = random.Random(rng_seed)

        # track touched registers (Unicorn reg ids) and touched 8-byte aligned addresses
        # Ensure PC, SP, and NZCV are always captured
        self._used_regs: Set[int] = {UC_ARM64_REG_PC, UC_ARM64_REG_SP, UC_ARM64_REG_NZCV}
        self._used_addresses: Set[int] = set()

        if arch == UC_ARCH_ARM64:
            self._encoder = AArch64Encoder()
        else:
            raise NotImplementedError("Only AArch64 supported for now")

        # Map initial memory region (pages); content left zeroed (lazy writes will overwrite)
        if initial_pages:
            self._data_base, self._data_size = initial_pages
        else:
            self._data_base = 0x1000_0000
            self._data_size = 0x2000

        self._uc.mem_map(self._data_base, self._data_size, UC_PROT_READ | UC_PROT_WRITE)

        self._instr_base = 0x2000_0000
        self._instr_size = 0x1_0000

        self._uc.mem_map(self._instr_base, self._instr_size, UC_PROT_READ | UC_PROT_EXEC)

    def _make_concrete(self, key: Union[str, int], template: InputTemplate) -> int:
        """
        key: either register name (str) or memory address (int)
        """
        val = self._value_strategy.choose_value(key, template)
        template.set_concrete(key, val)
        return val

    def _align_addr8(self, addr: int) -> int:
        return addr & (~0x7)

    def _mark_touched_addresses(self, addr: int) -> None:
        # If addr is aligned to 8 bytes, then lower_addr == upper_addr
        # else, lower_upper != upper_byte and therefore we want to mark both
        lower_byte = self._align_addr8(addr)
        upper_byte = self._align_addr8(addr + 7)
        self._used_addresses.add(lower_byte)
        self._used_addresses.add(upper_byte)

    def _init_src_operand(self, op: Operand, template: InputTemplate):
        """
        Ensure that a source operand has concrete data available in Unicorn before execution.
        """
        assert op.is_src

        if op.type == OT.REG:
            # op.value (str) used as InputTemplate key; op.reg_id (int) used for Unicorn
            reg_key = op.value
            reg_id = op.reg_id
            if not template.is_concrete(reg_key):
                val = self._make_concrete(reg_key, template)

                self._uc.reg_write(reg_id, val)
                self._used_regs.add(reg_id)

            assert reg_id in self._used_regs

        else:  # OT.MEM
            # represent memory keys in InputTemplate as integer addresses (Does not have to be 8 bytes aligned!)
            base_addr = self._data_base + op.offset
            self._mark_touched_addresses(base_addr)

            for address in range(base_addr, base_addr + 8):
                if not template.is_concrete(address):
                    val = self._make_concrete(address, template)
                    b = val.to_bytes(1, "little")
                    self._uc.mem_write(address, b)


    def execute_instruction(self, instr: Instruction, template: InputTemplate) -> ArchSimStep:
        # clone template so caller's template isn't polluted by speculative writes
        template = template.clone()

        operands = instr.operands + instr.implicit_operands

        # lazy concretization
        for op in operands:
            if op.is_src:
                self._init_src_operand(op, template)

        # encode instruction and write bytes
        asm = instr.to_asm_string()
        instr_addr = self._instr_base + instr.address
        raw_bytes = self._encoder.encode(asm, instr_addr)
        self._uc.mem_write(instr_addr, raw_bytes)

        # execute instruction
        self._uc.emu_start(instr_addr, instr_addr + len(raw_bytes))

        # No need to add to the InputTemplate, because it is a destination,
        # and therefore the input at this cell could be anything, we don't care
        for op in operands:
            if op.is_dest:
                if op.type == OT.REG:
                    self._used_regs.add(op.reg_id)

                else:  # OT.MEM
                    addr = self._data_base + op.offset
                    self._mark_touched_addresses(addr)
                    
        taken = None
        target = None
        if instr.control_flow:
            pc = self._uc.reg_read(UC_ARM64_REG_PC)
            taken = (pc != (instr_addr + len(raw_bytes)))
            target = pc

        return ArchSimStep(
            instruction_address=instr_addr,
            updated_input_template=template,
            taken=taken,
            target=target
        )

    def take_snapshot(self) -> UnicornSnapshot:
        return UnicornSnapshot(self, self._used_regs, self._used_addresses)

    def restore_snapshot(self, snap: UnicornSnapshot):
        snap.restore()

