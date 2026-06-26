"""PAC pointer-authentication seal and its sealing pass (AArch64).

A concrete Seal over the generic framework in aarch64_seal.py: genuine carries the correct
signature so authentication succeeds; a decoy carries a wrong signature (or strips the auth) so a
speculatively-authenticated value fails its check.
"""
from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from ..config import CONF
from ..interfaces import (Instruction, Operand, RegisterOperand, OT, MemoryOperand,
                          InstructionSpec, GeneratorException)
from .aarch64_seal import (Seal, FixPoint, SealedNIInstrumentation, _SandboxInstrumentationBase,
                           make_nop, index_instructions, _SANDBOX_MASK)
from .aarch64_target_desc import SANDBOX_BASE_REGISTER

if TYPE_CHECKING:
    from .aarch64_generator import Aarch64Generator

class PACKey(Enum):
    IA = 'ia'
    IB = 'ib'
    DA = 'da'
    DB = 'db'
    G  = 'g'

SLOT_SIG_POS  = 0   # slot position 0: NOP (stage-1) or MOVK upper-16 signature (TC1/TC2/TC3)
AUTH_SLOT_POS = 1   # slot position 1: XPAC (stage-1/TC1) or AUTH (TC2/TC3)
SLOT_SIZE     = 2

_PAC_INFO: Dict[str, Tuple[PACKey, str, str]] = {
    # pac_mnemonic: (key, auth_mnemonic, xpac_mnemonic)
    'pacia':  (PACKey.IA, 'autia',  'xpaci'),  'pacib':  (PACKey.IB, 'autib',  'xpaci'),
    'pacda':  (PACKey.DA, 'autda',  'xpacd'),  'pacdb':  (PACKey.DB, 'autdb',  'xpacd'),
    'paciza': (PACKey.IA, 'autiza', 'xpaci'),  'pacizb': (PACKey.IB, 'autizb', 'xpaci'),
    'pacdza': (PACKey.DA, 'autdza', 'xpacd'),  'pacdzb': (PACKey.DB, 'autdzb', 'xpacd'),
}
# Reverse maps derived from _PAC_INFO (keyed by auth mnemonic)
_AUTH_TO_KEY:  Dict[str, PACKey] = {auth: key  for _, (key, auth, _)    in _PAC_INFO.items()}
_AUTH_TO_XPAC: Dict[str, str]    = {auth: xpac for _, (_,   auth, xpac) in _PAC_INFO.items()}
_AUTH_TO_PAC:  Dict[str, str]    = {auth: pac  for pac, (_,  auth, _)   in _PAC_INFO.items()}


class PacSign(Seal):
    """PAC pointer-authentication seal.  Slot = [MOVK-signature, AUT*/XPAC*].

    Genuine carries the correct signature so authentication succeeds; a decoy carries an
    alternative signature so a speculatively-authenticated pointer fails its check. Single source
    of PAC slot encodings (the legacy PACInstrumentation and the engine both build through it). The
    signatures are read off the fix point; the seal does not know how they were resolved.
    """
    name = "pac_sign"
    slot_size = SLOT_SIZE

    def __init__(self, generator, auth_specs: Dict, xpac_specs: Dict):
        self.generator = generator
        self._auth_specs = auth_specs
        self._xpac_specs = xpac_specs

    # ---- low-level slot builders (single source of PAC slot encodings) ----
    def make_movk(self, reg: str, imm: int, lsl: int) -> Instruction:
        return Instruction("movk", True, "", False,
                           template=f"MOVK {reg}, #0x{imm & 0xFFFF:04x}, LSL #{lsl}")

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

    def _committed_info(self, fp) -> Tuple[str, str, str, Optional[str]]:
        """Return (auth_mn, xpac_mn, value_reg, ctx_reg): mnemonics/ctx from fp.committed_inst, the
        pointer register from fp.value_reg."""
        auth_mn = fp.committed_inst.name.lower()
        xpac_mn = _AUTH_TO_XPAC[auth_mn]
        ctx_reg = fp.committed_inst.operands[1].value if len(fp.committed_inst.operands) > 1 else None
        return auth_mn, xpac_mn, fp.value_reg, ctx_reg

    def _auth_fill(self, fp, sig: int) -> List[Instruction]:
        """[MOVK value_reg, #sig, LSL#48; AUTH value_reg, ctx_reg]."""
        auth_mn, _, value_reg, ctx_reg = self._committed_info(fp)
        return [self.make_movk(value_reg, sig, 48),
                self.make_auth_inst(auth_mn, value_reg, ctx_reg)]

    # ---- Seal protocol ----
    def placeholder(self, fp) -> List[Instruction]:
        """[NOP, XPAC] — strip, no auth yet."""
        _, xpac_mn, value_reg, _ = self._committed_info(fp)
        return [make_nop(), self.make_xpac_inst(xpac_mn, value_reg)]

    def genuine(self, fp) -> List[Instruction]:
        """[MOVK correct_sig LSL#48, AUTH] — auth succeeds; the placeholder when no signature was
        resolved."""
        if fp.correct_sig is None:
            return self.placeholder(fp)
        return self._auth_fill(fp, fp.correct_sig)

    def decoy(self, fp, rng: random.Random) -> List[Instruction]:
        """A wrong-signature forgery [MOVK alt, AUT*], alt drawn fresh from the pool so each decoy
        instance is a different forgery. Reached slots carry a PAC-mask-verified pool (provably fails
        AUTH); unreached slots carry a random pool (they never execute on the contract path)."""
        assert fp.alt_sigs, f"slot_id={fp.slot_id}: no decoy signatures resolved"
        return self._auth_fill(fp, rng.choice(fp.alt_sigs))


@dataclass
class PACFixPoint(FixPoint):
    committed_inst: Optional[Any] = None  # structural: the AUT* instruction the seal commits to
    # Per-input, recomputed by the executor from the CE trace (reset between inputs):
    correct_sig: Optional[int] = None        # upper-16 PAC bits signed by kernel; None if never reached
    alt_sigs: List[int] = field(default_factory=list)  # wrong upper-16 PAC bits, each differing from
    #                                  correct_sig within the PAC-field bits (provably fails AUTH); [] if unreached

    def reset(self) -> None:
        super().reset()
        self.correct_sig = None
        self.alt_sigs = []


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


class PACInstrumentation(_SandboxInstrumentationBase):

    def __init__(self, generator: Aarch64Generator, xpac_weight: int, auth_weight: int):
        xpac_weight = max(xpac_weight, 0)
        auth_weight = max(auth_weight, 0)
        total = (xpac_weight + auth_weight) * 1.0 or 1.0
        self._xpac_prob = xpac_weight / total
        self._auth_prob = auth_weight / total
        self.generator = generator
        self._norm = generator.target_desc.reg_normalized

        pac_instructions = [i for i in generator.instruction_set.instruction_unfiltered
                            if "PAC" in i.tags and
                            (CONF.supported_instructions is None or i.name in CONF.supported_instructions)]
        # Keep only instructions with 1 or 2 explicit operands where the first is a dest GPR.
        # This excludes: 0-operand system variants (pacia1716, paciasp…), 3-op pacga,
        # and src-only 1-op variants (autiasppcr, autibsppcr).
        def _is_usable_pac(i) -> bool:
            return 1 <= len(i.operands) <= 2 and i.operands[0].dest
        signing_instructions = list(filter(lambda i: i.name.lower().startswith('pac') and _is_usable_pac(i), pac_instructions))
        verification_instructions = list(filter(lambda i: i.name.lower().startswith('aut') and _is_usable_pac(i), pac_instructions))
        strip_sign_instructions  = list(filter(lambda i: i.name.lower().startswith('xpac') and _is_usable_pac(i), pac_instructions))


        self._pac_specs = {s.name.lower(): s for s in signing_instructions}
        self._auth_specs = {
            s.name.lower(): AuthInstructionSpec(
                s.name, s.category, s.control_flow, s.datatype,
                s.template, s.operands, s.implicit_operands, s.tags)
            for s in verification_instructions
        }
        self._xpac_specs = {s.name.lower(): s for s in strip_sign_instructions}
        self._seal = PacSign(generator, self._auth_specs, self._xpac_specs)
        # Sandbox parameters: mask lower bits of address and add sandbox base (x29).
        # Bundled with every signing operation so that signed values are always sandboxed.
        self._sandbox_mask = f"#0x{_SANDBOX_MASK:x}"
        self._sandbox_base_reg = SANDBOX_BASE_REGISTER

        # PAC/AUT/XPAC instructions are allowed in the base test case; the stage-1 pass
        # locates every AUT* and replaces it with an XPAC placeholder slot.

    # ------------------------------------------------------------------
    # Instruction builders
    # ------------------------------------------------------------------

    def _gen_distinct_operand_inst(self, specs: Dict, reg_operand: Optional[Operand] = None,
                                   modifier: Optional[Operand] = None) -> Instruction:
        """Generate an instruction from `specs` whose value_reg ≠ ctx_reg (zero-context variants
        are always accepted). Optionally force operand[0]=reg_operand and operand[1]=modifier."""
        for _ in range(20):
            instruction = self.generator.generate_instruction(random.choice(list(specs.values())))
            if reg_operand is not None and len(instruction.operands) >= 1:
                assert instruction.operands[0].type == OT.REG and reg_operand.type == OT.REG
                instruction.operands[0] = copy.deepcopy(reg_operand)
            if modifier is not None and len(instruction.operands) >= 2:
                assert instruction.operands[1].type == OT.REG and modifier.type == OT.REG
                instruction.operands[1] = copy.deepcopy(modifier)
            if len(instruction.operands) < 2:
                return instruction  # zero-context variant — always valid
            if self._norm_reg(instruction.operands[0].value) != \
                    self._norm_reg(instruction.operands[1].value):
                return instruction
        raise RuntimeError("Unable to generate a distinct-operand PAC/AUT instruction")

    def _get_signing_instruction(self, reg_operand: Optional[Operand] = None, modifier: Optional[Operand] = None) -> Instruction:
        return self._gen_distinct_operand_inst(self._pac_specs, reg_operand, modifier)

    def _get_auth_instruction(self) -> Instruction:
        """Generate a random AUT* instruction with distinct operand registers."""
        return self._gen_distinct_operand_inst(self._auth_specs)

    def _get_mem_auth_instruction(self, mem_reg: str) -> Optional[Instruction]:
        """Generate a random AUT* with value_reg forced to mem_reg and ctx_reg != mem_reg.

        Returns None if no valid instruction can be found (caller falls back to standalone sandbox).
        """
        norm_mem = self._norm_reg(mem_reg)
        for _ in range(20):
            candidate = self._get_auth_instruction()
            if len(candidate.operands) < 2:
                candidate.operands[0].value = mem_reg
                return candidate
            if self._norm_reg(candidate.operands[1].value) != norm_mem:
                candidate.operands[0].value = mem_reg
                return candidate
        return None

    # ------------------------------------------------------------------
    # Sealing pass: replace every AUT* with an XPAC placeholder slot;
    #               optionally insert a slot before memory accesses.
    # ------------------------------------------------------------------

    def _build_func_slots(
        self,
        func: Function,
        slot_counter: int,
        fix_points: List[PACFixPoint],
        auth_replacements: List,      # (old_auth_inst, bb, new_slot_insts)
        xpac_insertions: List,        # (mem_inst, bb, slot_insts, offset_subs, sandbox_insts)
        standalone_insertions: List,  # (mem_inst, bb, sandbox_insts + offset_subs)
    ) -> int:
        for bb in func:
            for inst in bb:
                mn = inst.name.lower()

                # AUT* in generated code: replace with [NOP, XPAC] slot
                if mn in self._auth_specs:
                    value_reg = inst.operands[0].value
                    # Disallow value_reg == ctx_reg: resample ctx from the same allowed value pool.
                    if len(inst.operands) > 1 and \
                            self._norm_reg(value_reg) == self._norm_reg(inst.operands[1].value):
                        norm_ptr = self._norm_reg(value_reg)
                        for _ in range(20):
                            fresh = self.generator.generate_instruction(self._auth_specs[mn])
                            if len(fresh.operands) > 1 and \
                                    self._norm_reg(fresh.operands[1].value) != norm_ptr:
                                inst.operands[1].value = fresh.operands[1].value
                                break
                    sid = slot_counter; slot_counter += 1
                    fp = PACFixPoint(slot_id=sid, value_reg=value_reg, committed_inst=copy.deepcopy(inst))
                    fp.slot_insts = self._seal.placeholder(fp)
                    fix_points.append(fp)
                    auth_replacements.append((inst, bb, fp.slot_insts))
                    continue  # don't also process as memory access

                # Memory access: sandbox + optional PAC slot
                if inst.has_memory_access:
                    if len(inst.get_mem_operands()) > 1:
                        raise GeneratorException(
                            "PAC instrumentation models one memory access per instruction; "
                            f"{inst.name!r} has several")
                    mem_reg = self._get_mem_base_reg(inst)
                    if mem_reg is None:
                        continue
                    offset_subs   = self._make_offset_sub_insts(inst.get_mem_operands()[0])
                    if self._is_tag_store(inst):   # STG-family: 16B-aligned clamp only, no PAC slot
                        standalone_insertions.append(
                            (inst, bb, self._make_sandbox_insts(mem_reg, align16=True) + offset_subs))
                        continue
                    sandbox_insts = self._make_sandbox_insts(mem_reg)
                    if self._auth_specs and random.random() < self._auth_prob:
                        auth_inst = self._get_mem_auth_instruction(mem_reg)
                        if auth_inst is not None:
                            sid = slot_counter; slot_counter += 1
                            fp = PACFixPoint(slot_id=sid, value_reg=auth_inst.operands[0].value,
                                             committed_inst=auth_inst)
                            fp.slot_insts = self._seal.placeholder(fp)
                            fix_points.append(fp)
                            xpac_insertions.append((inst, bb, fp.slot_insts, offset_subs, sandbox_insts))
                            continue
                    standalone_insertions.append((inst, bb, sandbox_insts + offset_subs))

        return slot_counter

    def seal_test_case(self, test_case: TestCase) -> Tuple[TestCase, List[PACFixPoint]]:
        """Seal the test case: replace AUT* with XPAC placeholder slots and add slots before some
        memory accesses. Returns (sealed_tc, fix_points)."""
        tc = copy.deepcopy(test_case)
        fix_points: List[PACFixPoint] = []
        slot_counter = 0

        for func in tc.functions:
            auth_replacements: List = []
            xpac_insertions: List = []
            standalone_insertions: List = []
            slot_counter = self._build_func_slots(
                func, slot_counter, fix_points,
                auth_replacements, xpac_insertions, standalone_insertions)

            # AUT* → [NOP, XPAC] (delete old instruction, insert 2 new ones)
            for old_inst, bb, new_insts in auth_replacements:
                for ni in new_insts:
                    bb.insert_before(old_inst, ni)
                bb.delete(old_inst)

            # Memory access with slot: [sandbox, NOP, XPAC, offset_subs, mem_access]
            for mem_inst, bb, slot_insts, offset_subs, sandbox_insts in xpac_insertions:
                for s in [*sandbox_insts, *slot_insts, *offset_subs]:
                    bb.insert_before(mem_inst, s)

            # Memory access without slot: [sandbox, offset_subs, mem_access]
            for mem_inst, bb, insts in standalone_insertions:
                for s in insts:
                    bb.insert_before(mem_inst, s)

        # Record each slot's position so the fills can locate it in any structural copy.
        locs = index_instructions(tc)
        for fp in fix_points:
            fp.slot_locs = [locs[id(si)] for si in fp.slot_insts]

        return tc, fix_points

    def make_engine(self, should_decoy=None) -> "SealedNIInstrumentation":
        """A non-interference engine driven by this pass's seal. Seal the test case with
        seal_test_case(), then feed the result to engine.set_sealed()."""
        return SealedNIInstrumentation(self._seal, should_decoy)
