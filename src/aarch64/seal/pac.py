"""PAC pointer-authentication slot encoders + signing (AArch64).

PacSign builds the MOVK-signature + AUT*/XPAC* slot instructions (the single source of PAC slot
encodings, used by the sealer's PacSealing); PacSigner is the kernel SIGN capability (never AUTH);
build_pac_specs derives the usable PAC/AUT*/XPAC instruction specs (AUT* via AuthInstructionSpec,
which guarantees value_reg != ctx_reg).
"""
import random
from typing import Dict, Optional, Tuple

from ...config import CONF
from ...interfaces import Instruction, InstructionSpec, RegisterOperand, ImmediateOperand


_PAC_INFO: Dict[str, Tuple[str, str]] = {
    # pac_mnemonic: (auth_mnemonic, xpac_mnemonic)
    'pacia':  ('autia',  'xpaci'),  'pacib':  ('autib',  'xpaci'),
    'pacda':  ('autda',  'xpacd'),  'pacdb':  ('autdb',  'xpacd'),
    'paciza': ('autiza', 'xpaci'),  'pacizb': ('autizb', 'xpaci'),
    'pacdza': ('autdza', 'xpacd'),  'pacdzb': ('autdzb', 'xpacd'),
}
# Reverse maps derived from _PAC_INFO (keyed by auth mnemonic)
_AUTH_TO_XPAC: Dict[str, str] = {auth: xpac for auth, xpac in _PAC_INFO.values()}
_AUTH_TO_PAC:  Dict[str, str] = {auth: pac  for pac, (auth, _) in _PAC_INFO.items()}


class PacSign:
    """Builds PAC slot encodings — the MOVK-signature + AUT*/XPAC* instructions. The single source of
    PAC slot instruction encoders, used by the sealer's PacSealing."""

    def __init__(self, generator, auth_specs: Dict, xpac_specs: Dict):
        self.generator = generator
        self._auth_specs = auth_specs
        self._xpac_specs = xpac_specs

    # ---- low-level slot builders (single source of PAC slot encodings) ----
    def make_movk(self, reg: str, imm: int, lsl: int) -> Instruction:
        imm &= 0xFFFF
        return (Instruction("movk", True, "", False, template=f"MOVK {reg}, #0x{imm:04x}, LSL #{lsl}")
                .add_op(RegisterOperand(reg, 64, False, True))
                .add_op(ImmediateOperand(str(imm), 16))
                .add_op(ImmediateOperand(str(lsl), 6)))

    def make_xpac_inst(self, mnemonic: str, reg: str) -> Instruction:
        inst = self.generator.generate_instruction(self._xpac_specs[mnemonic])
        inst.operands[0].value = reg
        return inst

    def make_auth_inst(self, mnemonic: str, reg: str, ctx_reg: Optional[str]) -> Instruction:
        inst = self.generator.generate_instruction(self._auth_specs[mnemonic])
        inst.operands[0].value = reg
        if ctx_reg is not None and len(inst.operands) > 1:
            inst.operands[1].value = ctx_reg
        return inst

    def _pick_mem_auth(self, value_reg: str) -> Instruction:
        """A random AUT* over value_reg with ctx_reg != value_reg — the authentication chosen for a
        memory pointer that has no generator-emitted AUT* of its own."""
        norm = self.generator.target_desc.reg_normalized
        norm_base = norm.get(value_reg, value_reg)
        for _ in range(20):
            inst = self.generator.generate_instruction(self._auth_specs[random.choice(list(self._auth_specs))])
            inst.operands[0].value = value_reg
            if len(inst.operands) < 2 or \
                    norm.get(inst.operands[1].value, inst.operands[1].value) != norm_base:
                return inst
        raise RuntimeError(f"cannot pick a memory AUT* for {value_reg}")


def _read_reg(cpu, reg: Optional[str]) -> Optional[int]:
    if reg is None:
        return None
    return cpu.sp if reg == "sp" else cpu.gpr[int(reg[1:])]


class PacSigner:
    """The kernel PAC signing capability — the only PAC code that touches the hardware keys. Pure
    SIGN, never AUTH (a failed AUTH at EL1 resets the box). `field_mask16` characterizes which top-16
    bits SIGN sets (the PAC field), so a forgery confined to them provably fails AUTH."""

    def __init__(self, pac_sign):
        self._pac_sign = pac_sign       # local_executor.pac_sign(ptr, ctx, mnemonic) -> signed 64-bit
        self._mask_cache: Dict[str, int] = {}

    def sign16(self, ptr: int, ctx: int, mn: str) -> int:
        return (self._pac_sign(ptr, ctx, mn) >> 48) & 0xFFFF

    def field_mask16(self, mn: str, samples: int = 64) -> int:
        m = self._mask_cache.get(mn)
        if m is None:
            mask = 0
            for _ in range(samples):
                v = random.randrange(1 << 48)
                mask |= self._pac_sign(v, random.randrange(1 << 64), mn) ^ v
            m = (mask >> 48) & 0xFFFF
            assert m and not (m & 0x80), f"implausible PAC-field mask 0x{m:04x}"
            self._mask_cache[mn] = m
        return m


class AuthInstructionSpec(InstructionSpec):
    """AUT* instruction spec that retries generation until value_reg ≠ ctx_reg."""

    def generate(self, generator) -> Instruction:
        norm = generator.target_desc.reg_normalized
        for _ in range(20):
            inst = super().generate(generator)
            if len(inst.operands) < 2:
                return inst  # zero-context variant — always valid
            if norm.get(inst.operands[0].value) != norm.get(inst.operands[1].value):
                return inst
        raise RuntimeError(f"Cannot generate {self.name} with value_reg ≠ ctx_reg")


def build_pac_specs(generator) -> Tuple[Dict, Dict, Dict]:
    """(pac_specs, auth_specs, xpac_specs) for PAC/AUT*/XPAC instructions usable as 1-2 operand
    dest-GPR ops (excludes 0-operand system variants, 3-op pacga, src-only variants)."""
    pac_instructions = [i for i in generator.instruction_set.instruction_unfiltered
                        if "PAC" in i.tags and
                        (CONF.supported_instructions is None or i.name in CONF.supported_instructions)]
    def _usable(i) -> bool:
        return 1 <= len(i.operands) <= 2 and i.operands[0].dest
    pac_specs = {s.name.lower(): s for s in pac_instructions
                 if s.name.lower().startswith('pac') and _usable(s)}
    auth_specs = {s.name.lower(): AuthInstructionSpec(s.name, s.category, s.control_flow, s.datatype,
                  s.template, s.operands, s.implicit_operands, s.tags)
                  for s in pac_instructions if s.name.lower().startswith('aut') and _usable(s)}
    xpac_specs = {s.name.lower(): s for s in pac_instructions
                  if s.name.lower().startswith('xpac') and _usable(s)}
    return pac_specs, auth_specs, xpac_specs
