"""
File: AArch64 test case generator
"""
import abc
import os
import math
import random
import copy
import functools
from itertools import chain
from subprocess import Popen, PIPE
from typing import Any, List, Tuple, Optional, Set, Dict, Iterator, Callable
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


from .seal.primitives import _SANDBOX_MASK, _SandboxInstrumentationBase


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
        # B.AL / B.NV both branch "always" on A64, so they are additional unconditional forms
        template, name = random.choice((
            ("B {label}", "b"),
            ("B.al {label}", "b.al"),
            ("B.nv {label}", "b.nv"),
        ))
        return Instruction(name, False, "UNCOND_BR", True, template=template)

    @staticmethod
    def assemble(asm_file: str, obj_file: str, bin_file: str) -> None:
        """Assemble an AArch64 test case into a stripped flat binary (cross GNU as/objcopy)."""
        ConfigurableGenerator._assemble(asm_file, obj_file, bin_file,
                                        "aarch64-linux-gnu-as -march=armv9-a+sve+memtag",
                                        "aarch64-linux-gnu-objcopy")

    @staticmethod
    def in_memory_assemble(asm: str) -> bytes:
        """Assemble AArch64 assembly to raw machine code via the asm_to_bytes helper.

        Assembly is a pure function of the (normalized) source text, so the result
        is memoized: the sealed/NI fuzzing path re-assembles the same placeholder
        test case once per input (measured: ~72% of calls are byte-identical), and
        each call spawns three processes (asm_to_bytes -> as + objcopy). The cache
        turns those redundant spawns into dict hits."""
        if not asm.endswith('\n'):
            asm += '\n'
        return Aarch64Generator._assemble_cached(asm)

    @staticmethod
    @functools.lru_cache(maxsize=2048)
    def _assemble_cached(asm: str) -> bytes:
        asm_to_bytes = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "asm_to_bytes", "asm_to_bytes")
        p = Popen([asm_to_bytes], stdin=PIPE, stdout=PIPE, stderr=PIPE, text=False)
        machine_code, err = p.communicate(asm.encode("ascii"))

        if p.returncode != 0:
            raise RuntimeError(f"asm_to_bytes failed:\n{err.decode()}")

        return machine_code

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

# AUT* with an explicit context register (Xd = pointer/dest, Xn = context). Xn == Xd is unsatisfiable
# once sealed: the seal rewrites AUT* to [MOVK sig, AUT*], and the MOVK writes the signature into
# Xd[63:48] before the AUT* reads Xd as its context, so the auth context (sig|addr) never matches the
# clean addr the sig was computed over → the genuine AUT* FPAC-faults. Force ctx ≠ ptr. (PAC* sign
# instructions are fine: they only sign, never fault, and the seal does not rewrite them.)
_AUTH_CTX = frozenset({"autia", "autib", "autda", "autdb"})


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

        elif name in _AUTH_CTX:
            self._patch_auth_context_collision(inst)

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

    def _patch_auth_context_collision(self, inst: Instruction) -> None:
        """Force an AUT*'s context register (Xn) to differ from its pointer register (Xd); see _AUTH_CTX."""
        ptr, ctx = inst.operands[0], inst.operands[1]
        assert isinstance(ptr, RegisterOperand) and isinstance(ctx, RegisterOperand)
        if self._norm(ptr.value) == self._norm(ctx.value):
            self._replace_reg(ctx, forbidden={self._norm(ptr.value)})

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
        """Replace operand.value with a register of the same file and width that is not in
        *forbidden*. The letter prefix fixes the file+width, so candidates keep it (q stays q,
        d stays d, x stays x) -- a SIMD operand is never replaced by a GPR or re-named v<->q."""
        prefix = operand.value.rstrip("0123456789")
        if prefix[0] in ("x", "w") or prefix == "sp":
            pool = self.target_desc.registers.get(operand.width, [])
        else:
            pool = [f"{prefix}{i}" for i in range(32)]
        candidates = [r for r in pool if self._norm(r) not in forbidden]
        if not candidates:
            # Should not happen in a correctly configured target descriptor,
            # but guard against it rather than crashing the fuzzer.
            raise RuntimeError("unable to solve constraints! unexpected!")

        operand.value = random.choice(candidates)


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
        align16 = self._is_tag_store(instr)   # STG-family needs a 16-byte-aligned address
        for mem_op in mem_ops:
            base = self._base_reg(mem_op)
            for inst in self._make_sandbox_insts(base, align16) + self._make_offset_sub_insts(mem_op):
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

    def generate_cond_operand(self, spec: OperandSpec, inst: Instruction) -> Operand:
        values = spec.values
        if inst.control_flow:
            # AL/NV branch unconditionally; keep a conditional branch's fallthrough reachable.
            values = [c for c in values if c not in ("al", "nv")]
        return CondOperand(random.choice(values))

    def generate_mem_operand(self, spec: MemorySpec, inst: Instruction) -> Operand:
        # a memory access wraps its address components (base/index/offset/extend); generate each via the
        # normal dispatch (which also stamps its MemoryRole), and keep them as the operand's `inner`.
        inner = [self.generate_operand(component, inst) for component in spec.inner]
        address = ", ".join(op.value for op in inner)
        return MemoryOperand(address, spec.width, spec.src, spec.dest, inner=inner)
