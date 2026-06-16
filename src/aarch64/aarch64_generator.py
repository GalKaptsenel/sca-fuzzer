"""
File: AArch64 test case generator
"""
import abc
import math
import random
import copy
from itertools import chain
from typing import Any, List, Tuple, Optional, Set, Dict
from dataclasses import dataclass, field
from enum import Enum, auto

from ..config import CONF
from ..isa_loader import InstructionSet
from ..interfaces import TestCase, Operand, Instruction, BasicBlock, Function, InstructionSpec, \
    GeneratorException, RegisterOperand, MAIN_AREA_SIZE, FAULTY_AREA_SIZE, \
    MemoryOperand, OT, OperandSpec, MemorySpec, CondOperand
from ..generator import ConfigurableGenerator, RandomGenerator, Pass
from .aarch64_target_desc import Aarch64TargetDesc, SANDBOX_BASE_REGISTER, AArch64MemRole
from .aarch64_elf_parser import Aarch64ElfParser
from .aarch64_printer import Aarch64Printer

_SANDBOX_MASK_BITS = int(math.log(MAIN_AREA_SIZE + FAULTY_AREA_SIZE, 2))
_SANDBOX_MASK = (1 << _SANDBOX_MASK_BITS) - 1


class Aarch64Generator(ConfigurableGenerator, abc.ABC):
    target_desc: Aarch64TargetDesc

    def __init__(self, instruction_set: InstructionSet, seed: int):
        super(Aarch64Generator, self).__init__(instruction_set, seed)
        self.target_desc = Aarch64TargetDesc()
        self.elf_parser = Aarch64ElfParser(self.target_desc)

        self.passes = [
            Aarch64PatchUndefinedLoadsStoresPass(self.target_desc),
        ]

        self.printer = Aarch64Printer(self.target_desc)

    def get_return_instruction(self) -> Instruction:
        return Instruction("ret", False, "", True, template="RET")

    def get_unconditional_jump_instruction(self) -> Instruction:
        return Instruction("b", False, "UNCOND_BR", True, template="B {label}")

    def get_elf_data(self, test_case: TestCase, obj_file: str) -> None:
        self.elf_parser.parse(test_case, obj_file)


class Aarch64DsbSyPass(Pass):

    def run_on_test_case(self, test_case: TestCase) -> None:
        for func in test_case.functions:
            for bb in func:
                insertion_points = []
                for instr in bb:
                    # make a copy to avoid infinite insertions
                    insertion_points.append(instr)

                for instr in insertion_points:
                    bb.insert_after(instr, Instruction("DSB SY", True, template="DSB SY"))


# Single-dest load with base writeback (post/pre-index).
# dest == base → UNPREDICTABLE.
_LDR_WRITEBACK = frozenset({
    "ldr", "ldrb", "ldrh", "ldrsb", "ldrsh", "ldrsw",
})

# Two-dest load, all address forms.
# dest0 == dest1 → UNPREDICTABLE.
# Post/pre-index additionally: dest0|dest1 == base → UNPREDICTABLE.
_LDP_ANY = frozenset({"ldp", "ldpsw"})

# Exclusive two-dest load (always has a base, no explicit writeback field,
# but the base register must not alias either destination).
_LDXP = frozenset({"ldxp", "ldaxp"})

# Store-exclusive: status register must not alias data or base.
_STXR = frozenset({"stxr", "stlxr", "stxrb", "stlxrb", "stxrh", "stlxrh"})

# Store-exclusive pair: same as STXR plus data0==data1 check on status.
_STXP = frozenset({"stxp", "stlxp"})

# Store pair with writeback: src0|src1 == base → UNPREDICTABLE.
_STP_WRITEBACK = frozenset({"stp"})

# Single-register store with writeback: transferred reg == base → UNPREDICTABLE.
_STR_WRITEBACK = frozenset({"str", "strb", "strh"})


class Aarch64PatchUndefinedLoadsStoresPass(Pass):
    """
    Patch all UNPREDICTABLE register-collision constraints for AArch64
    load and store instructions.
    """

    def __init__(self, target_desc) -> None:
        self.target_desc: Aarch64TargetDesc = target_desc
        super().__init__()

    # ------------------------------------------------------------------
    # Pass entry point
    # ------------------------------------------------------------------

    def run_on_test_case(self, test_case: TestCase) -> None:
        for func in test_case.functions:
            for bb in func:
                for inst in bb:
                    self._patch_instruction(inst)

    def _patch_instruction(self, inst: Instruction) -> None:
        # sandboxing subtracts the index from the base (SUB base, base, index), so base == index would
        # zero the base and the access would escape the sandbox; force them to differ first.
        self._patch_address_register_collision(inst)

        name = inst.name.lower()

        if any(name.startswith(m) for m in _LDR_WRITEBACK):
            self._patch_single_writeback(inst)

        elif any(name.startswith(m) for m in _STR_WRITEBACK):
            self._patch_single_writeback(inst)

        elif any(name.startswith(m) for m in _LDP_ANY):
            self._patch_ldp(inst)

        elif any(name.startswith(m) for m in _LDXP):
            self._patch_ldxp(inst)

        elif any(name.startswith(m) for m in _STXR):
            self._patch_stxr(inst)

        elif any(name.startswith(m) for m in _STXP):
            self._patch_stxp(inst)

        elif any(name.startswith(m) for m in _STP_WRITEBACK):
            self._patch_stp_writeback(inst)

    def _patch_address_register_collision(self, inst: Instruction) -> None:
        """For every memory access, force the index register to differ from the base. The sandbox masks
        the base then subtracts the index (`SUB base, base, index`); if they are the same register the
        base becomes zero and the effective address escapes the sandbox."""
        for mem_op in inst.get_mem_operands():
            base = next((c for c in mem_op.inner if c.mem_role is AArch64MemRole.BASE), None)
            index = next((c for c in mem_op.inner if c.mem_role is AArch64MemRole.INDEX), None)
            if base is not None and index is not None \
                    and self._norm(base.value) == self._norm(index.value):
                self._replace_reg(index, forbidden={self._norm(base.value)})

    def _writes_back(self, inst: Instruction) -> bool:
        """Whether the access updates its base register (pre/post-index), marked on the inner base."""
        return any(c.mem_role is AArch64MemRole.BASE and c.dest
                   for mem_op in inst.get_mem_operands() for c in mem_op.inner)

    def _patch_single_writeback(self, inst: Instruction) -> None:
        # Single-register load/store with writeback: the transferred register (ops[0]) must differ
        # from the base; t == n is CONSTRAINED UNPREDICTABLE.
        if not self._writes_back(inst):
            return  # unsigned-offset or register-offset form — no writeback

        ops = inst.operands
        if len(ops) < 2:
            return

        assert isinstance(ops[0], RegisterOperand)
        assert isinstance(ops[1], MemoryOperand)

        norm_dest = self._norm(ops[0].value)
        if norm_dest in self._mem_regs_normalized(ops[1]):
            self._replace_reg(ops[0], forbidden={norm_dest} | self._mem_regs_normalized(ops[1]))

    def _patch_ldp(self, inst: Instruction) -> None:
        ops = inst.operands
        if len(ops) < 3:
            return

        assert isinstance(ops[0], RegisterOperand)
        assert isinstance(ops[1], RegisterOperand)
        assert isinstance(ops[2], MemoryOperand)

        has_writeback = self._writes_back(inst)
        norm0 = self._norm(ops[0].value)
        norm1 = self._norm(ops[1].value)
        base_norms = self._mem_regs_normalized(ops[2])

        # Constraint: dest0 != dest1
        if norm0 == norm1:
            self._replace_reg(ops[1], forbidden={norm1})
            norm1 = self._norm(ops[1].value)  # refresh after patch

        if has_writeback:
            # Constraint: dest0 not in base
            if norm0 in base_norms:
                self._replace_reg(ops[0], forbidden=base_norms | {norm1})
                norm0 = self._norm(ops[0].value)

            # Constraint: dest1 not in base
            if norm1 in base_norms:
                self._replace_reg(ops[1], forbidden=base_norms | {norm0})

    def _patch_ldxp(self, inst: Instruction) -> None:
        ops = inst.operands
        if len(ops) < 3:
            return

        assert isinstance(ops[0], RegisterOperand)
        assert isinstance(ops[1], RegisterOperand)
        assert isinstance(ops[2], MemoryOperand)

        base_norms = self._mem_regs_normalized(ops[2])
        norm0 = self._norm(ops[0].value)
        norm1 = self._norm(ops[1].value)

        if norm0 == norm1:
            self._replace_reg(ops[1], forbidden={norm1})
            norm1 = self._norm(ops[1].value)

        if norm0 in base_norms:
            self._replace_reg(ops[0], forbidden=base_norms | {norm1})
            norm0 = self._norm(ops[0].value)

        if norm1 in base_norms:
            self._replace_reg(ops[1], forbidden=base_norms | {norm0})

    def _patch_stp_writeback(self, inst: Instruction) -> None:
        if not self._writes_back(inst):
            return  # unsigned-offset form — no writeback constraint

        ops = inst.operands
        if len(ops) < 3:
            return

        assert isinstance(ops[0], RegisterOperand)
        assert isinstance(ops[1], RegisterOperand)
        assert isinstance(ops[2], MemoryOperand)

        base_norms = self._mem_regs_normalized(ops[2])

        if self._norm(ops[0].value) in base_norms:
            self._replace_reg(ops[0], forbidden=base_norms)

        if self._norm(ops[1].value) in base_norms:
            self._replace_reg(ops[1], forbidden=base_norms)

    def _patch_stxr(self, inst: Instruction) -> None:
        ops = inst.operands
        if len(ops) < 3:
            return

        assert isinstance(ops[0], RegisterOperand)  # Ws (status)
        assert isinstance(ops[1], RegisterOperand)  # Rt (data)
        assert isinstance(ops[2], MemoryOperand)

        norm_status = self._norm(ops[0].value)
        norm_data = self._norm(ops[1].value)
        base_norms = self._mem_regs_normalized(ops[2])

        forbidden = {norm_data} | base_norms
        if norm_status in forbidden:
            self._replace_reg(ops[0], forbidden=forbidden)

    def _patch_stxp(self, inst: Instruction) -> None:
        ops = inst.operands
        if len(ops) < 4:
            return

        assert isinstance(ops[0], RegisterOperand)  # Ws (status)
        assert isinstance(ops[1], RegisterOperand)  # Rt1
        assert isinstance(ops[2], RegisterOperand)  # Rt2
        assert isinstance(ops[3], MemoryOperand)

        norm_status = self._norm(ops[0].value)
        forbidden = {
            self._norm(ops[1].value),
            self._norm(ops[2].value),
        } | self._mem_regs_normalized(ops[3])

        if norm_status in forbidden:
            self._replace_reg(ops[0], forbidden=forbidden)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _norm(self, reg: str) -> str:
        """Normalise a register name through the target descriptor."""
        return self.target_desc.reg_normalized[reg]

    def _mem_regs_normalized(self, mem_op: MemoryOperand) -> Set[str]:
        """Normalised names of the registers used in the address: the base and any index register
        (the offset/extend components are immediates, not registers)."""
        return {self._norm(op.value) for op in mem_op.inner if op.type == OT.REG}

    def _replace_reg(self, operand: RegisterOperand, forbidden: Set[str]) -> None:
        """
        Replace operand.value with a randomly chosen register of the same
        width that is not in *forbidden* (compared after normalisation).
        """
        candidates = [
            r for r in self.target_desc.registers[operand.width]
            if self._norm(r) not in forbidden
        ]
        if not candidates:
            # Should not happen in a correctly configured target descriptor,
            # but guard against it rather than crashing the fuzzer.
            raise RuntimeError("unable to solve constraints! unexpected!")

        operand.value = random.choice(candidates)


class PACVariant(Enum):
    """Labels for the TC variants produced by PAC stage-2 instrumentation."""
    STRIP_ONLY    = auto()  # XPAC strip before every access — no auth, arch-safe baseline
    AUTH_CORRECT  = auto()  # AUTIA with correct signed pointer and context (auth always succeeds)
    AUTH_WRONG    = auto()  # AUTIA with wrong PAC bits/ctx for spec slots — NI test

class MTEVariant(Enum):
    """Labels for the TC variants produced by MTE stage-2 instrumentation."""
    BASELINE      = auto()  # NOP placeholders — no tag change, reference behavior
    RANDOMIZE_TAG = auto()  # IRG on spec accesses — random tag (correct for NI contract)
    WRONG_TAG     = auto()  # MOVK wrong upper16 for spec accesses — deterministic wrong tag

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


@dataclass
class PACFixPoint:
    slot_id: int
    slot_insts: List[Instruction] = field(default_factory=list)
    committed_inst: Optional[Any] = None  # AUT* Instruction committed at stage-1 build time
    # Per-input values populated by executor from CE trace (reset between inputs):
    spec_nesting: Optional[int] = None
    correct_sig: Optional[int] = None  # upper-16 PAC bits signed by kernel; None if CE never reached
    alt_sig:     Optional[int] = None  # upper-16 PAC bits from an alternative (ptr/ctx) combo; for TC3 spec

    def reset(self) -> None:
        self.spec_nesting = None
        self.correct_sig  = None
        self.alt_sig      = None


class _SandboxInstrumentationBase:
    """Shared helpers for sandbox-taint-based instrumentation passes."""

    _norm: Dict[str, str]
    _sandbox_mask: str
    _sandbox_base_reg: str

    def _norm_reg(self, reg: str) -> str:
        return self._norm.get(reg, reg)

    def _dest_regs(self, inst: Instruction) -> frozenset:
        result = {self._norm_reg(op.value) for op in inst.operands + inst.implicit_operands
                  if op.dest and op.type == OT.REG and op.value in self._norm}
        # writeback (pre/post-index) updates the base register; the extractor marks that on the inner
        # base component (component.dest), so read it directly instead of guessing from operand names.
        for mem_op in inst.get_mem_operands():
            for c in mem_op.inner:
                if c.dest and c.type == OT.REG and c.value in self._norm:
                    result.add(self._norm_reg(c.value))
        return frozenset(result)

    @staticmethod
    def _base_reg(mem_op: MemoryOperand) -> str:
        for c in mem_op.inner:
            if c.mem_role is AArch64MemRole.BASE:
                return c.value
        raise GeneratorException(f"memory operand {mem_op.value!r} has no base register")

    def _get_mem_base_reg(self, inst: Instruction) -> Optional[str]:
        mem_ops = inst.get_mem_operands()
        return self._base_reg(mem_ops[0]) if mem_ops else None

    def _make_offset_sub_insts(self, mem_op: MemoryOperand) -> List[Instruction]:
        """SUBs that remove every non-base contribution to the address (an index register with its
        optional shift/extend, or a standalone displacement) from the base, so the effective address
        lands exactly at the base's already-sandboxed value. Components carry their roles."""
        base = self._base_reg(mem_op)
        comp = {c.mem_role: c for c in mem_op.inner}
        index = comp.get(AArch64MemRole.INDEX)
        offset = comp.get(AArch64MemRole.OFFSET)
        extend = comp.get(AArch64MemRole.EXTEND)
        if index is not None:
            # EA = base + <shift/extend>(index); replicate the modifier in the SUB so it cancels exactly
            if extend is not None:                          # extended register: e.g. `, sxtw #2`
                modifier = f", {extend.value}" + (f" #{offset.value}" if offset is not None else "")
            elif offset is not None:                        # shifted register: `, lsl #amount`
                modifier = f", lsl #{offset.value}"
            else:
                modifier = ""
            return [Instruction("sub", True, "", False,
                                template=f"SUB {base}, {base}, {index.value}{modifier}")]
        if offset is not None:                              # standalone displacement
            return self._make_disp_sub_insts(base, int(offset.value))
        return []

    def _make_disp_sub_insts(self, base: str, disp: int) -> List[Instruction]:
        """Cancel a signed displacement from base (chunked to the 12-bit immediate), so [base+disp]
        resolves to base. A negative displacement is cancelled with ADD."""
        mnemonic = "SUB" if disp >= 0 else "ADD"
        remaining, result = abs(disp), []
        while remaining > 0:
            chunk = min(4095, remaining)
            result.append(Instruction(mnemonic.lower(), True, "", False,
                                       template=f"{mnemonic} {base}, {base}, #{chunk}"))
            remaining -= chunk
        return result

    def _make_sandbox_insts(self, reg: str) -> List[Instruction]:
        """Return [AND reg, reg, #mask; ADD reg, reg, x29] that sandbox reg into the input region."""
        and_inst = Instruction("and", True, "", False, template=f"AND {reg}, {reg}, {self._sandbox_mask}")
        add_inst = Instruction("add", True, "", False, template=f"ADD {reg}, {reg}, {self._sandbox_base_reg}")
        return [and_inst, add_inst]

    @staticmethod
    def _topo_sort(func: Function) -> Tuple[Dict[BasicBlock, List[BasicBlock]], List[BasicBlock]]:
        """Return (predecessors, topo_order) for func's CFG."""
        predecessors: Dict[BasicBlock, List[BasicBlock]] = {}
        for bb in func:
            predecessors.setdefault(bb, [])
            for succ in bb.successors:
                predecessors.setdefault(succ, []).append(bb)
        topo: List[BasicBlock] = []
        seen: Set[BasicBlock] = set()
        def _dfs(bb: BasicBlock) -> None:
            if bb in seen:
                return
            seen.add(bb)
            for succ in bb.successors:
                _dfs(succ)
            topo.append(bb)
        _dfs(func.get_first_bb())
        topo.reverse()
        return predecessors, topo


class AuthInstructionSpec(InstructionSpec):
    """AUT* instruction spec that retries generation until ptr_reg ≠ ctx_reg."""

    def generate(self, generator) -> Instruction:
        norm = generator.target_desc.reg_normalized
        for _ in range(20):
            inst = super().generate(generator)
            if len(inst.operands) < 2:
                return inst  # zero-context variant — always valid
            if norm.get(inst.operands[0].value) != norm.get(inst.operands[1].value):
                return inst
        raise RuntimeError(f"Cannot generate {self.name} with ptr_reg ≠ ctx_reg")


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
                            if "BASE-PAC" in i.tags and
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
        """Generate an instruction from `specs` whose ptr_reg ≠ ctx_reg (zero-context variants
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
        """Generate a random AUT* with ptr_reg forced to mem_reg and ctx_reg != mem_reg.

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

    def _make_auth_inst(self, mnemonic: str, reg: str, ctx_reg: Optional[str]) -> Instruction:
        inst = self.generator.generate_instruction(self._auth_specs[mnemonic])
        inst.operands[0].value = reg
        if ctx_reg is not None and len(inst.operands) > 1:
            inst.operands[1].value = ctx_reg
        return inst

    def _make_xpac_inst(self, mnemonic: str, reg: str, slot_id: int, pos: int) -> Instruction:
        inst = self.generator.generate_instruction(self._xpac_specs[mnemonic])
        inst.operands[0].value = reg
        inst._pac_slot_id = slot_id
        inst._pac_slot_pos = pos
        return inst

    def _make_nop(self, slot_id: int, pos: int) -> Instruction:
        nop = Instruction("nop", True, "", False, template="NOP")
        nop._pac_slot_id = slot_id
        nop._pac_slot_pos = pos
        return nop

    def _make_movk(self, slot_id: int, pos: int, reg: str, imm: int, lsl: int) -> Instruction:
        inst = Instruction("movk", True, "", False, template=f"MOVK {reg}, #0x{imm & 0xFFFF:04x}, LSL #{lsl}")
        inst._pac_slot_id = slot_id
        inst._pac_slot_pos = pos
        return inst

    # ------------------------------------------------------------------
    # Stage 1: replace every AUT* with an XPAC placeholder slot;
    #          optionally insert a slot before memory accesses.
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
                    xpac_mn = _AUTH_TO_XPAC[mn]
                    ptr_reg = inst.operands[0].value
                    # Disallow ptr_reg == ctx_reg: resample ctx from the same allowed value pool.
                    if len(inst.operands) > 1 and \
                            self._norm_reg(ptr_reg) == self._norm_reg(inst.operands[1].value):
                        norm_ptr = self._norm_reg(ptr_reg)
                        for _ in range(20):
                            fresh = self.generator.generate_instruction(self._auth_specs[mn])
                            if len(fresh.operands) > 1 and \
                                    self._norm_reg(fresh.operands[1].value) != norm_ptr:
                                inst.operands[1].value = fresh.operands[1].value
                                break
                    sid = slot_counter; slot_counter += 1
                    slot_insts = [
                        self._make_nop(sid, SLOT_SIG_POS),
                        self._make_xpac_inst(xpac_mn, ptr_reg, sid, AUTH_SLOT_POS),
                    ]
                    fix_points.append(PACFixPoint(slot_id=sid, slot_insts=slot_insts,
                                               committed_inst=copy.deepcopy(inst)))
                    auth_replacements.append((inst, bb, slot_insts))
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
                    sandbox_insts = self._make_sandbox_insts(mem_reg)
                    offset_subs   = self._make_offset_sub_insts(inst.get_mem_operands()[0])
                    if self._auth_specs and random.random() < self._auth_prob:
                        auth_inst = self._get_mem_auth_instruction(mem_reg)
                        if auth_inst is not None:
                            xpac_mn = _AUTH_TO_XPAC[auth_inst.name.lower()]
                            sid = slot_counter; slot_counter += 1
                            slot_insts = [
                                self._make_nop(sid, SLOT_SIG_POS),
                                self._make_xpac_inst(xpac_mn, mem_reg, sid, AUTH_SLOT_POS),
                            ]
                            fix_points.append(PACFixPoint(slot_id=sid, slot_insts=slot_insts,
                                                       committed_inst=auth_inst))
                            xpac_insertions.append((inst, bb, slot_insts, offset_subs, sandbox_insts))
                            continue
                    standalone_insertions.append((inst, bb, sandbox_insts + offset_subs))

        return slot_counter

    def instrument_stage1(self, test_case: TestCase) -> Tuple[TestCase, List[PACFixPoint]]:
        """Replace AUT* instructions with XPAC placeholder slots; add slots before some memory accesses.

        Returns (instrumented_tc, fix_points).
        """
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

            _slot_ids_now = {
                i._pac_slot_id
                for _f in tc.functions for _b in _f for i in _b
                if hasattr(i, '_pac_slot_id')
            }
            for fp in fix_points:
                assert fp.slot_id in _slot_ids_now, (
                    f"stage1 insertion bug: slot_id={fp.slot_id} missing from tc after insertions"
                )

        return tc, fix_points

    # ------------------------------------------------------------------
    # Stage 2: produce TC1 / TC2 / TC3 from stage-1 result
    # ------------------------------------------------------------------

    def _committed_info(self, fp: PACFixPoint) -> Tuple[str, str, str, Optional[str]]:
        """Return (auth_mn, xpac_mn, ptr_reg, ctx_reg) from fp.committed_inst."""
        auth_mn = fp.committed_inst.name.lower()
        xpac_mn = _AUTH_TO_XPAC[auth_mn]
        ptr_reg = fp.committed_inst.operands[0].value
        ctx_reg = fp.committed_inst.operands[1].value if len(fp.committed_inst.operands) > 1 else None
        return auth_mn, xpac_mn, ptr_reg, ctx_reg

    def _make_tc1_slot(self, fp: PACFixPoint) -> List[Instruction]:
        """TC1 (STRIP_ONLY): [MOVK correct_sig LSL#48, XPAC] if sig known, else [NOP, XPAC]."""
        _, xpac_mn, ptr_reg, _ = self._committed_info(fp)
        sig_inst = (
            self._make_movk(fp.slot_id, SLOT_SIG_POS, ptr_reg, fp.correct_sig, 48)
            if fp.correct_sig is not None
            else self._make_nop(fp.slot_id, SLOT_SIG_POS)
        )
        return [sig_inst, self._make_xpac_inst(xpac_mn, ptr_reg, fp.slot_id, AUTH_SLOT_POS)]

    def _make_tc2_slot(self, fp: PACFixPoint) -> List[Instruction]:
        """TC2 (AUTH_CORRECT): [MOVK ptr_reg, #correct_sig, LSL#48, AUTH ptr_reg, ctx_reg]."""
        auth_mn, _, ptr_reg, ctx_reg = self._committed_info(fp)
        assert fp.correct_sig is not None  # caller must guard: only call when CE reached the slot
        movk = self._make_movk(fp.slot_id, SLOT_SIG_POS, ptr_reg, fp.correct_sig, 48)
        auth = self._make_auth_inst(auth_mn, ptr_reg, ctx_reg)
        auth._pac_slot_id  = fp.slot_id
        auth._pac_slot_pos = AUTH_SLOT_POS
        return [movk, auth]

    def _make_tc3_spec_slot(self, fp: PACFixPoint) -> List[Instruction]:
        """TC3 spec: [MOVK ptr_reg, #alt_sig, LSL#48, AUTH ptr_reg, ctx_reg].

        alt_sig is an alternative signing combination (different ptr and/or ctx). Always set by executor.
        """
        auth_mn, _, ptr_reg, ctx_reg = self._committed_info(fp)
        assert fp.alt_sig is not None, f"slot_id={fp.slot_id}: alt_sig not set by executor"
        movk = self._make_movk(fp.slot_id, SLOT_SIG_POS, ptr_reg, fp.alt_sig, 48)
        auth = self._make_auth_inst(auth_mn, ptr_reg, ctx_reg)
        auth._pac_slot_id  = fp.slot_id
        auth._pac_slot_pos = AUTH_SLOT_POS
        return [movk, auth]

    def instrument_stage2(
        self, prep_tc: TestCase, fix_points: List[PACFixPoint],
    ) -> Dict[PACVariant, TestCase]:
        """Produce TC1/TC2/TC3 variants from the stage-1 TC.

        TC1 (STRIP_ONLY)   — [MOVK correct_sig, XPAC]     — same strip as stage-1, sig pre-loaded
        TC2 (AUTH_CORRECT) — [MOVK correct_sig, AUTH]     — AUTH always succeeds
        TC3 (AUTH_WRONG)   — arch slots: same as TC2
                           — spec slots: [MOVK alt_sig, AUTH]
        If CE never reached a slot (correct_sig is None): all variants use TC1.
        """
        strip_tc   = copy.deepcopy(prep_tc)
        correct_tc = copy.deepcopy(prep_tc)
        wrong_tc   = copy.deepcopy(prep_tc)
        maps = {
            PACVariant.STRIP_ONLY:   self._find_slot_insts(strip_tc),
            PACVariant.AUTH_CORRECT: self._find_slot_insts(correct_tc),
            PACVariant.AUTH_WRONG:   self._find_slot_insts(wrong_tc),
        }

        for fp in fix_points:
            self._fill_slot(maps[PACVariant.STRIP_ONLY], fp, self._make_tc1_slot(fp))
            if fp.correct_sig is None:
                # CE never reached this slot: CORRECT → TC1 (safe), WRONG → alt_sig
                self._fill_slot(maps[PACVariant.AUTH_CORRECT], fp, self._make_tc1_slot(fp))
                self._fill_slot(maps[PACVariant.AUTH_WRONG],   fp, self._make_tc3_spec_slot(fp))
                continue
            # spec_nesting is None when CE never reached the slot → treat as speculative
            is_spec = fp.spec_nesting != 0 if fp.spec_nesting is not None else True
            self._fill_slot(maps[PACVariant.AUTH_CORRECT], fp, self._make_tc2_slot(fp))
            if not is_spec:
                self._fill_slot(maps[PACVariant.AUTH_WRONG], fp, self._make_tc2_slot(fp))
            else:
                self._fill_slot(maps[PACVariant.AUTH_WRONG], fp, self._make_tc3_spec_slot(fp))

        return {
            PACVariant.STRIP_ONLY:   strip_tc,
            PACVariant.AUTH_CORRECT: correct_tc,
            PACVariant.AUTH_WRONG:   wrong_tc,
        }

    def _find_slot_insts(self, tc: TestCase) -> Dict[int, Dict[int, Tuple[Instruction, BasicBlock]]]:
        """Walk tc and return {slot_id: {pos: (inst, bb)}} for all tagged slot instructions."""
        slot_map: Dict[int, Dict[int, Tuple[Instruction, BasicBlock]]] = {}
        for func in tc.functions:
            for bb in func:
                for inst in bb:
                    if hasattr(inst, '_pac_slot_id'):
                        sid: int = inst._pac_slot_id
                        pos: int = inst._pac_slot_pos
                        slot_map.setdefault(sid, {})[pos] = (inst, bb)
        return slot_map

    def _fill_slot(self, slot_map: Dict, fp: PACFixPoint, new_insts: List[Instruction]) -> None:
        """Replace the SLOT_SIZE instructions in slot_map[fp.slot_id] with new_insts (padded with NOPs)."""
        positions = slot_map.get(fp.slot_id)
        assert positions is not None, (
            f"slot_id={fp.slot_id} not found in slot_map "
            f"(slot_map keys={sorted(slot_map.keys())})"
        )
        for pos in range(SLOT_SIZE):
            old_inst, bb = positions[pos]
            new_inst = new_insts[pos] if pos < len(new_insts) else self._make_nop(fp.slot_id, pos)
            bb.insert_before(old_inst, new_inst)
            bb.delete(old_inst)


# ===========================================================================
# MTE non-interference instrumentation
# ===========================================================================

MTE_SLOT_SIZE = 1  # one NOP placeholder per memory access


@dataclass
class MTEFixPoint:
    """Per-memory-access metadata for MTE stage-2 variant generation."""
    slot_id: int
    bb: BasicBlock
    mem_inst: Instruction
    reg: str                       # original base register name
    slot_insts: List[Instruction]  # single-element: [nop placeholder]
    spec_nesting: Optional[int] = None

    def reset(self) -> None:
        self.spec_nesting = None


class MTEInstrumentation(_SandboxInstrumentationBase):
    """
    Two-stage MTE non-interference instrumentation.

    Stage 1: insert a NOP placeholder before every memory access.
             For registers not yet correctly sandbox-tagged, AND+ADD is prepended.
             Taint = frozenset of normalized register names holding a correctly-
             sandbox-tagged address (via AND+ADD with x29, not subsequently overwritten).
             Taint is cleared on any register write; intersection at CFG join nodes.
             ADDG/SUBG with imm4==0 propagate the source tag to the destination.

    Stage 2: replace each NOP placeholder to produce TC1/TC2/TC3:
             TC1 → NOP (correct flow, arch_tag everywhere — baseline)
             TC2 → arch: NOP;  spec: IRG Xd,Xd  (random tag)
             TC3 → arch: NOP;  spec: MOVK Xd,#wrong_upper16,LSL#48  (deterministic wrong tag)
    """

    def __init__(self, generator: Aarch64Generator):
        self.generator = generator
        self._norm = generator.target_desc.reg_normalized
        self._sandbox_mask = f"#0x{_SANDBOX_MASK:x}"
        self._sandbox_base_reg = SANDBOX_BASE_REGISTER
        self.last_taint_log: List[str] = []

    # ------------------------------------------------------------------
    # Instruction builders
    # ------------------------------------------------------------------

    def _make_mte_nop(self, slot_id: int) -> Instruction:
        nop = Instruction("nop", True, "", False, template="NOP")
        nop._mte_slot_id = slot_id
        return nop

    def _make_mte_irg(self, reg: str, slot_id: int) -> Instruction:
        inst = Instruction("irg", True, "", False, template=f"IRG {reg}, {reg}")
        inst._mte_slot_id = slot_id
        return inst

    def _make_mte_movk_wrong_tag(self, reg: str, wrong_upper16: int, slot_id: int) -> Instruction:
        inst = Instruction("movk", True, "", False,
                           template=f"MOVK {reg}, #0x{wrong_upper16 & 0xFFFF:04x}, LSL #48")
        inst._mte_slot_id = slot_id
        return inst

    # ------------------------------------------------------------------
    # Taint helpers
    # ------------------------------------------------------------------

    def _mte_tag_propagates(self, inst: Instruction) -> Optional[Tuple[str, str]]:
        """If inst preserves the tag unchanged (ADDG/SUBG with imm4==0),
        return (dest_reg, src_reg). Otherwise None."""
        if inst.name.lower() not in ('addg', 'subg'):
            return None
        if len(inst.operands) < 4:
            return None
        try:
            imm4 = int(inst.operands[3].value)
        except (ValueError, IndexError):
            return None
        if imm4 != 0:
            return None
        return inst.operands[0].value, inst.operands[1].value

    # ------------------------------------------------------------------
    # Stage-1 dataflow pass
    # ------------------------------------------------------------------

    def _build_mte_slots(
        self,
        func: Function,
        slot_counter: int,
        fix_points: List,
        insertions: List,
        taint_log: List,
    ) -> int:
        """Topological taint pass: build NOP-placeholder fix_points for every memory access.

        taint (curr) = frozenset of normalized register names that hold a
        correctly-sandbox-tagged address (via AND+ADD with x29).
        Cleared on any register write; intersection at CFG join nodes.
        ADDG/SUBG with imm4==0 propagate the source tag to the destination.

        For each memory access:
          tainted base  → offset_subs + NOP placeholder only
          untainted base → AND+ADD + offset_subs + NOP placeholder; reg added to taint
        """
        predecessors, topo = self._topo_sort(func)
        taint_out: Dict[BasicBlock, frozenset] = {}

        for bb in topo:
            processed = [p for p in predecessors.get(bb, []) if p in taint_out]
            if not processed:
                curr: frozenset = frozenset()
            elif len(processed) == 1:
                curr = taint_out[processed[0]]
            else:
                curr = taint_out[processed[0]]
                for p in processed[1:]:
                    curr = curr & taint_out[p]

            for inst in bb:
                if inst.has_memory_access:
                    if len(inst.get_mem_operands()) > 1:
                        raise GeneratorException(
                            "MTE instrumentation models one memory access per instruction; "
                            f"{inst.name!r} has several")
                    mem_reg = self._get_mem_base_reg(inst)
                    if mem_reg is not None:
                        norm_mem = self._norm_reg(mem_reg)
                        offset_subs = self._make_offset_sub_insts(inst.get_mem_operands()[0])
                        sid = slot_counter
                        slot_counter += 1
                        nop = self._make_mte_nop(sid)
                        fp = MTEFixPoint(slot_id=sid, bb=bb, mem_inst=inst,
                                         reg=mem_reg, slot_insts=[nop])
                        fix_points.append(fp)

                        if norm_mem in curr:
                            insertions.append((inst, bb, [], offset_subs, [nop]))
                            taint_log.append(
                                f"  MEM-ACCESS   inst={inst.name:12s}  base={mem_reg}"
                                f"  decision=NOP-ONLY"
                                f"  taint={sorted(curr)}")
                        else:
                            sandbox_insts = self._make_sandbox_insts(mem_reg)
                            insertions.append((inst, bb, sandbox_insts, offset_subs, [nop]))
                            curr = curr | frozenset([norm_mem])
                            taint_log.append(
                                f"  MEM-ACCESS   inst={inst.name:12s}  base={mem_reg}"
                                f"  decision=SANDBOX+NOP"
                                f"  taint={sorted(curr)}")
                    else:
                        taint_log.append(
                            f"  MEM-ACCESS   inst={inst.name:12s}  base=None(implicit?)"
                            f"  decision=SKIP")

                prop = self._mte_tag_propagates(inst)
                for dreg in self._dest_regs(inst):
                    if prop is not None and dreg == self._norm_reg(prop[0]):
                        norm_src = self._norm_reg(prop[1])
                        if norm_src in curr:
                            curr = curr | frozenset([dreg])
                            taint_log.append(
                                f"  TAG-PRESERVE inst={inst.name:12s}  {norm_src}->{dreg}")
                        else:
                            if dreg in curr:
                                taint_log.append(
                                    f"  ARITH-CLEAR  inst={inst.name:12s}  clears={dreg}")
                            curr = curr - frozenset([dreg])
                    else:
                        if dreg in curr:
                            taint_log.append(
                                f"  ARITH-CLEAR  inst={inst.name:12s}  clears={dreg}")
                        curr = curr - frozenset([dreg])

            taint_out[bb] = curr

        return slot_counter

    # ------------------------------------------------------------------
    # Stage 1 public API
    # ------------------------------------------------------------------

    def instrument_stage1(self, test_case: TestCase) -> Tuple[TestCase, List[MTEFixPoint]]:
        """Instrument test_case with NOP placeholders before every memory access.

        Returns (instrumented_tc, fix_points).
        instrumented_tc contains:
          - optional AND+ADD before each memory access (for untainted registers)
          - optional offset SUBs (for immediate-offset addressing modes)
          - NOP placeholder immediately before each memory access
        """
        tc = copy.deepcopy(test_case)
        fix_points: List[MTEFixPoint] = []
        slot_counter = 0
        self.last_taint_log = []

        for func in tc.functions:
            insertions: List = []
            func_log: List[str] = []
            slot_counter = self._build_mte_slots(
                func, slot_counter, fix_points, insertions, func_log)
            self.last_taint_log.extend(func_log)

            for mem_inst, bb, sandbox_insts, offset_subs, nop_insts in insertions:
                for s in sandbox_insts:
                    bb.insert_before(mem_inst, s)
                for s in offset_subs:
                    bb.insert_before(mem_inst, s)
                for s in nop_insts:
                    bb.insert_before(mem_inst, s)

        return tc, fix_points

    # ------------------------------------------------------------------
    # Stage-2 slot helpers
    # ------------------------------------------------------------------

    def _find_slot_insts(self, tc: TestCase) -> Dict[int, Tuple[Any, BasicBlock]]:
        """Return {slot_id: (inst, bb)} for all MTE-tagged instructions in tc."""
        slot_map: Dict[int, Tuple[Any, BasicBlock]] = {}
        for func in tc.functions:
            for bb in func:
                for inst in bb:
                    if hasattr(inst, '_mte_slot_id'):
                        slot_map[inst._mte_slot_id] = (inst, bb)
        return slot_map

    def _fill_slot(self, slot_map: Dict, fp: MTEFixPoint, new_inst: Instruction) -> None:
        """Replace the NOP placeholder for fp.slot_id with new_inst."""
        entry = slot_map.get(fp.slot_id)
        assert entry is not None, (
            f"MTE slot_id={fp.slot_id} not in slot_map "
            f"(keys={sorted(slot_map.keys())})")
        old_inst, bb = entry
        bb.insert_before(old_inst, new_inst)
        bb.delete(old_inst)

    # ------------------------------------------------------------------
    # Stage 2 public API
    # ------------------------------------------------------------------

    def instrument_stage2(
        self,
        prep_tc: TestCase,
        fix_points: List[MTEFixPoint],
        sandbox_base: int,
    ) -> Dict[MTEVariant, TestCase]:
        """Produce TC1/TC2/TC3 from the stage-1 instrumented TC.

        sandbox_base is used to compute the deterministic wrong tag for TC3.
        spec_nesting must be populated on each PACFixPoint before calling.

        TC1 — correct flow: all placeholders → NOP
        TC2 — arch: NOP;  spec: IRG Xd,Xd  (randomizes bits[59:56])
        TC3 — arch: NOP;  spec: MOVK Xd,#wrong_upper16,LSL#48
                   wrong_upper16 preserves bits[63:60] and [55:48] of sandbox_base,
                   and sets bits[59:56] (tag) to arch_tag XOR 1.
        """
        sandbox_upper16 = (sandbox_base >> 48) & 0xFFFF
        arch_tag = (sandbox_base >> 56) & 0xF
        wrong_tag = arch_tag ^ 1
        wrong_upper16 = (sandbox_upper16 & ~(0xF << 8)) | (wrong_tag << 8)

        baseline_tc   = copy.deepcopy(prep_tc)
        randomize_tc  = copy.deepcopy(prep_tc)
        wrong_tag_tc  = copy.deepcopy(prep_tc)
        maps = {
            MTEVariant.BASELINE:      self._find_slot_insts(baseline_tc),
            MTEVariant.RANDOMIZE_TAG: self._find_slot_insts(randomize_tc),
            MTEVariant.WRONG_TAG:     self._find_slot_insts(wrong_tag_tc),
        }

        for fp in fix_points:
            # spec_nesting=0    → arch path → RANDOMIZE_TAG=NOP, WRONG_TAG=NOP (correct tag preserved)
            # spec_nesting>0    → spec path → RANDOMIZE_TAG=IRG, WRONG_TAG=MOVK wrong tag
            # spec_nesting=None → CE never executed this memory access (non-arch path);
            #                     hardware may speculate on it → treat as spec
            is_spec = fp.spec_nesting != 0

            self._fill_slot(maps[MTEVariant.BASELINE], fp, self._make_mte_nop(fp.slot_id))

            if is_spec:
                self._fill_slot(maps[MTEVariant.RANDOMIZE_TAG], fp, self._make_mte_irg(fp.reg, fp.slot_id))
            else:
                self._fill_slot(maps[MTEVariant.RANDOMIZE_TAG], fp, self._make_mte_nop(fp.slot_id))

            if is_spec:
                self._fill_slot(maps[MTEVariant.WRONG_TAG], fp,
                                self._make_mte_movk_wrong_tag(fp.reg, wrong_upper16, fp.slot_id))
            else:
                self._fill_slot(maps[MTEVariant.WRONG_TAG], fp, self._make_mte_nop(fp.slot_id))

        return {
            MTEVariant.BASELINE:      baseline_tc,
            MTEVariant.RANDOMIZE_TAG: randomize_tc,
            MTEVariant.WRONG_TAG:     wrong_tag_tc,
        }


class Aarch64SandboxPass(Pass, _SandboxInstrumentationBase):
    def __init__(self):
        super().__init__()
        self._sandbox_mask = f"#0x{_SANDBOX_MASK:x}"
        self._sandbox_base_reg = SANDBOX_BASE_REGISTER

    def run_on_test_case(self, test_case: TestCase) -> None:
        for func in test_case.functions:
            for bb in func:
                mem_instructions = [inst for inst in bb if inst.has_mem_operand(True)]
                for inst in mem_instructions:
                    self.sandbox_memory_access(inst, bb)

    def sandbox_memory_access(self, instr: Instruction, parent: BasicBlock) -> None:
        """Force every memory access of *instr* into the sandbox region. For each memory operand, mask
        its base into [x29 .. x29+mask], then cancel the index/offset/extend so the effective address
        lands at the masked base. Multiple memory operands (e.g. MOPS copy) are each handled.

        Limitation: the mask bounds the base, not base+access_size, so a multi-byte access (notably a
        16-byte LDP/STP) whose masked base is within the last few bytes of the region can spill past
        it. Rare and pre-existing; widen the mask to (region - max_access_size) if it ever matters."""
        if instr.get_implicit_mem_operands():
            raise GeneratorException("Implicit memory accesses are not supported")
        mem_ops = instr.get_mem_operands()
        if not mem_ops:
            raise GeneratorException("Attempt to sandbox an instruction without memory operands")
        for mem_op in mem_ops:
            base = self._base_reg(mem_op)
            for inst in self._make_sandbox_insts(base) + self._make_offset_sub_insts(mem_op):
                parent.insert_before(instr, inst)


class Aarch64RandomGenerator(Aarch64Generator, RandomGenerator):

    def __init__(self, instruction_set: InstructionSet, seed: int):
        super().__init__(instruction_set, seed)

    def _filter_invalid_operands(self, spec: OperandSpec, inst: Instruction) -> List[str]:
        result: List[str] = []
        register_prefixes = ("x", "w", "q", "v", "d", "s", "h", "b", "sp")

        for op in spec.values:
            if 'pc' == op:
                result.append(op)
            elif not op.startswith(register_prefixes):
                result.append(op)
            elif op in chain.from_iterable(self.target_desc.registers.values()):
                # avoid the same physical register being both here and inside a memory operand of this
                # instruction (the assembler warns on / rejects such forms). The address registers are
                # the inner components of the memory access (base/index), not the operand's value string.
                mem_regs = [c.value for m in inst.operands if isinstance(m, MemoryOperand)
                            for c in m.inner if c.type == OT.REG]
                if any(v not in Aarch64TargetDesc.reg_normalized for v in mem_regs):
                    continue
                if all(Aarch64TargetDesc.reg_normalized[op] != Aarch64TargetDesc.reg_normalized[v]
                       for v in mem_regs):
                    result.append(op)
        return result

    def generate_reg_operand(self, spec: OperandSpec, inst: Instruction) -> Operand:
        choices = self._filter_invalid_operands(spec, inst)
        reg = random.choice(choices)
        return RegisterOperand(reg, spec.width, spec.src, spec.dest)

    def generate_cond_operand(self, spec: OperandSpec, _: Instruction) -> Operand:
        cond = random.choice(spec.values)
        return CondOperand(cond)

    def generate_mem_operand(self, spec: MemorySpec, inst: Instruction) -> Operand:
        # a memory access wraps its address components (base/index/offset/extend); generate each via the
        # normal dispatch (which also stamps its MemoryRole), and keep them as the operand's `inner`.
        inner = [self.generate_operand(component, inst) for component in spec.inner]
        address = ", ".join(op.value for op in inner)
        return MemoryOperand(address, spec.width, spec.src, spec.dest, inner=inner)
