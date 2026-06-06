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
import unicorn.arm64_const as ucc

from .input_template import InputTemplate
from .value_selector import ValueSelectionStrategy
from .arch_simulator import ArchSimStep, ArchSimulatorInterface, ArchSnapshotInterface
from ..interfaces import Instruction, Operand, OT
from ..model import UnicornTargetDesc
from ..aarch64.aarch64_target_desc import Aarch64TargetDesc


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
        'pc': ucc.UC_ARM64_REG_PC,
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
        self._uc.reg_write(UC_ARM64_REG_PC, self._instr_base)

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
        instr_addr = self._uc.reg_read(UC_ARM64_REG_PC)
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


        instr_addr = self._uc.reg_read(UC_ARM64_REG_PC)
#        print(f'Running {instr.to_asm_string()} at address 0x{instr_addr:X}')
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
        raw_bytes = self._encoder.encode(asm, instr_addr)
        self._uc.mem_write(instr_addr, raw_bytes)

        # execute instruction and log memory write operations
        ea_logs = []
        hook = self._uc.hook_add(UC_HOOK_MEM_WRITE, self._mem_hook, user_data=ea_logs)
        self._uc.emu_start(instr_addr, 0, count=1)
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

