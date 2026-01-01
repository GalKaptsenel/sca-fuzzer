from __future__ import annotations
import warnings
import random

from typing import Optional, Set, Tuple, Union
from unicorn import (
    Uc, UC_ARCH_ARM64, UC_MODE_ARM, UC_PROT_READ, UC_PROT_WRITE, UC_PROT_EXEC, UC_HOOK_MEM_READ, UC_HOOK_MEM_WRITE, UC_HOOK_MEM_UNMAPPED, UC_MEM_READ, UC_MEM_WRITE
)
from unicorn import *
from unicorn.arm64_const import (
    UC_ARM64_REG_PC, UC_ARM64_REG_SP, UC_ARM64_REG_NZCV, UC_ARM64_REG_X30
)
from keystone import Ks, KS_ARCH_ARM64, KS_MODE_LITTLE_ENDIAN

from .input_template import InputTemplate
from .value_selector import ValueSelectionStrategy
from .arch_simulator import ArchSimStep, ArchSimulatorInterface, ArchSnapshotInterface
from ..interfaces import Instruction, Operand, OT
from ..aarch64.aarch64_target_desc import Aarch64UnicornTargetDesc, Aarch64TargetDesc


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
            addr &= ~0x7
            self._memory[addr] = self._uc.mem_read(addr, 8)

    def restore(self):
        self._uas._used_regs = self._to_restore_regs.copy()
        self._uas._used_addresses = self._to_restore_addresses.copy()

        for r, val in self._regs.items():
            self._uc.reg_write(r, int(val))

        for addr, data in self._memory.items():
            self._uc.mem_write(addr, bytes(data))

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
      - op.src / op.dest booleans indicate source/destination roles
    Assumptions about InputTemplate:
      - template.is_concrete(key) accepts either str (reg name) or int (address)
      - template.set_concrete(key, value) accepts same keys
      - keys for memory are integer addresses (does not have to be aligned!)

    Notice:
        For snapshot we are aligning to 8 bytes, and snapshot the entire byte
    """

    def hook_mem_unmapped(self, uc, access, address, size, value, user_data):
        import pdb; pdb.set_trace()
        print(f"[UNMAPPED] X30={uc.reg_read(UC_ARM64_REG_X30)} PC={uc.reg_read(UC_ARM64_REG_PC)}")
        if access == UC_MEM_READ_UNMAPPED:
            print(f"[UNMAPPED READ] addr=0x{address:x}, size={size}")
        elif access == UC_MEM_WRITE_UNMAPPED:
            print(f"[UNMAPPED WRITE] addr=0x{address:x}, size={size}, value=0x{value:x}")
        elif access == UC_MEM_FETCH_UNMAPPED:
            print(f"[UNMAPPED FETCH] addr=0x{address:x}")
        else:
            print(f"[UNMAPPED ?] access={access}, addr=0x{address:x}")
    
        return False  # stop emulation → Unicorn raises UC_ERR_READ/WRITE_UNMAPPED


    def __init__(
        self,
        arch: int = UC_ARCH_ARM64,
        mode: int = UC_MODE_ARM,
        value_strategy: ValueSelectionStrategy = None,
        rng_seed: Optional[int] = None,
        initial_pages: Optional[Tuple[int, int]] = None
    ):
        if value_strategy is None:
            raise RuntimeError("ValueSelectionStrategy required")
        self._uc = Uc(arch, mode)
        self._value_strategy = value_strategy
        self._rnd = random.Random(rng_seed)

        # track touched registers (Unicorn reg ids) and touched 8-byte aligned addresses
        # Ensure PC, SP, and NZCV are always captured
        self._used_regs: Set[int] = {UC_ARM64_REG_PC, UC_ARM64_REG_SP, UC_ARM64_REG_NZCV, UC_ARM64_REG_X30}
        self._used_addresses: Set[int] = set()

        if arch == UC_ARCH_ARM64:
            self._encoder = AArch64Encoder()
        else:
            raise NotImplementedError("Only AArch64 supported for now")

        PAGE_SIZE = 0x1000
        ZERO_PAGE = b"\x00" * PAGE_SIZE

        # Map initial memory region (pages); content left zeroed (lazy writes will overwrite)
        if initial_pages:
            self._data_base, self._data_size = initial_pages
        else:
            self._data_base = 0x1000_0000
            self._data_size = 2 * PAGE_SIZE

        DATA_SIZE_WITH_OVERFLOW_PAGE = self._data_size + PAGE_SIZE
        self._uc.mem_map(self._data_base, DATA_SIZE_WITH_OVERFLOW_PAGE, UC_PROT_READ | UC_PROT_WRITE)
        for page in range(self._data_base, DATA_SIZE_WITH_OVERFLOW_PAGE, PAGE_SIZE):
            self._uc.mem_write(page, ZERO_PAGE)

        # TODO: Workaround - For now, we statically decide that x30, in arm64 is used for storing the data_base value
        self._uc.reg_write(UC_ARM64_REG_X30, int(self._data_base))

        self._uc.hook_add(
                UC_HOOK_MEM_UNMAPPED,
                self.hook_mem_unmapped
            )

        self._instr_base = 0x2000_0000
        self._instr_size = 10 * PAGE_SIZE

        self._uc.mem_map(self._instr_base, self._instr_size, UC_PROT_READ | UC_PROT_EXEC)

    def _make_concrete(self, key: Union[str, int], template: InputTemplate) -> int:
        """
        key: either register name (str) or memory address (int)
        """
        val = self._value_strategy(key, template)
        template.get_cell(key).set_concrete(val)
        return int(val)

    def _align_addr8(self, addr: int) -> int:
        return addr & (~0x7)

    def _mark_touched_addresses(self, addr: int, length: int) -> None:
        assert length > 0
        lower_byte = self._align_addr8(addr)
        upper_byte = self._align_addr8(addr + length - 1)
        for b in range(lower_byte, upper_byte + 1, 8):
            self._used_addresses.add(b)

    def _snapshot_regs_save(self, regs_to_snapshot: List[str]) -> Dict[str, int]:
        snapshot = {}
        for r in regs_to_snapshot:
            reg_id_to_snapshot = Aarch64UnicornTargetDesc.reg_decode[Aarch64TargetDesc.reg_normalized[r]]
            snapshot[r] = self._uc.reg_read(reg_id_to_snapshot)

        return snapshot

    def _snapshot_regs_restore(self, regs_snapshot: Dict[str, int]) -> None:
        for r, val in regs_snapshot.items():
            reg_id_to_restore = Aarch64UnicornTargetDesc.reg_decode[Aarch64TargetDesc.reg_normalized[r]]
            self._uc.reg_write(reg_id_to_restore, int(val))

    def _mem_hook(self, uc, access, address, size, value, user_data):
        user_data.append({
            "ea": address,
            "size": size,
            "type": "READ" if access == UC_MEM_READ else "WRITE",
            "value": value,
            })

    def _init_src_reg(self, op: RegisterOperand, template: InputTemplate):
        """
        Ensure that a source operand has concrete data available in Unicorn before execution.
        """
        assert op.src
        if op.value not in self._used_regs:
            return

        # op.value (str) used as InputTemplate key; op.reg_id (int) used for Unicorn
        reg_key = op.value
        reg_id = Aarch64UnicornTargetDesc.reg_decode[Aarch64TargetDesc.reg_normalized[reg_key]]

        if not template.get_cell(reg_key).is_concrete():
            val = self._make_concrete(reg_key, template)

            self._uc.reg_write(reg_id, val)
            self._used_regs.add(reg_id)

        assert reg_id in self._used_regs

    def _compute_src_eas(self, instr: Instruction) -> List["EA entries"]:
        regs_to_snapshot = [
                f'x{i}' for i in range(31)
            ] + ['nzcv', 'sp', 'pc']

        eas_log = []
        snapshot = self._snapshot_regs_save(regs_to_snapshot)
        hook = self._uc.hook_add(UC_HOOK_MEM_READ, self._mem_hook, user_data=eas_log)
        asm = instr.to_asm_string()
        instr_addr = self._instr_base + 0 #instr.address
        raw_bytes = self._encoder.encode(asm, instr_addr)
        self._uc.mem_write(instr_addr, raw_bytes)
        self._uc.emu_start(instr_addr, instr_addr + len(raw_bytes))
        self._uc.hook_del(hook)
        self._snapshot_regs_restore(snapshot)
        return eas_log

    def _mark_touched_eas(self, ea_logs: List["EA entries"]):
        for entry in ea_logs:
            self._mark_touched_addresses(entry['ea'], entry['size'])

    def _init_src_memory(self, memory_to_initialize: List["EA entries"], template: InputTemplate):
        self._mark_touched_eas(memory_to_initialize)
        for entry in memory_to_initialize:
            assert entry['type'] == "READ"
            lower_byte = self._align_addr8(entry['ea'])
            upper_byte = self._align_addr8(entry['ea'] + entry['size'] - 1)
            for byte_address in range(lower_byte, upper_byte + 1, 8):

                if byte_address >= self._data_base + self._data_size:
                    break

                sandbox_byte_offset = byte_address - self._data_base
                if not template.get_cell(sandbox_byte_offset).is_concrete():
                    val = self._make_concrete(sandbox_byte_offset, template)
                    b = int(val).to_bytes(8, "little")
                    self._uc.mem_write(byte_address, b)

    def execute_instruction(self, instr: Instruction, template: InputTemplate) -> ArchSimStep:
        # clone template so caller's template isn't polluted by speculative writes
        template = template.clone()


        print(f'Running {instr.to_asm_string()}')
        # lazy concretization
        initialize_memory = False
        for op in instr.get_src_operands(include_implicit=True):
            if op.type == OT.MEM:
                initialize_memory = True
            if op.value in Aarch64TargetDesc.reg_normalized:
                self._init_src_reg(op, template)
        if initialize_memory:
            ea_logs = self._compute_src_eas(instr)
            self._init_src_memory(ea_logs, template)

        # encode instruction and write bytes
        asm = instr.to_asm_string()
        instr_addr = self._instr_base + 0 #instr.address
        raw_bytes = self._encoder.encode(asm, instr_addr)
        self._uc.mem_write(instr_addr, raw_bytes)

        # execute instruction and log memory write operations
        ea_logs = []
        hook = self._uc.hook_add(UC_HOOK_MEM_WRITE, self._mem_hook, user_data=ea_logs)
        self._uc.emu_start(instr_addr, 0, count=1) #instr_addr + len(raw_bytes))
        self._uc.hook_del(hook)

        # No need to add to the InputTemplate, because it is a destination,
        # and therefore the input at this cell could be anything, we don't care
        self._mark_touched_eas(ea_logs)
        for op in instr.get_dest_operands(include_implicit=True):
            if op.value in Aarch64TargetDesc.reg_normalized:
                self._used_regs.add(Aarch64UnicornTargetDesc.reg_decode[Aarch64TargetDesc.reg_normalized[op.value]])
                    
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

