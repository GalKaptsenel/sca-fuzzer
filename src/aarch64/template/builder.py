"""The template surface: a `Template.build` emits a program through the `Builder` -- fixed instructions,
holes, and calls, all as typed objects (no assembly strings), validated against the spec at build time."""
from __future__ import annotations

import abc
import contextlib
import copy
import random
from typing import Callable, List, Optional, Union

from ...config import CONF
from ...interfaces import (Instruction, OT, RegisterOperand, LabelOperand, Function, BasicBlock,
                           DispatchTable)
from ..aarch64_generator import Aarch64IndirectCallPass, Aarch64PatchUndefinedLoadsStoresPass
from ..aarch64_target_desc import INDIRECT_CALL_TARGET_REGISTER, AArch64MemRole
from .objects import Count, Reg, Mem, Imm, Hole, Populate, Call, DirectCall, IndirectCall, Cond
from .specs import SpecRef, resolve


def resolve_count(n: Count) -> int:
    """Concrete count for a fill: an exact int, a uniform draw from a (min, max) range, or -- for None
    -- a random count up to the configured program size. Deterministic under the seeded RNG."""
    if isinstance(n, int):
        return n
    if isinstance(n, tuple):
        return random.randint(n[0], n[1])
    return random.randint(1, max(1, CONF.program_size))


class HoleInstruction(Instruction):
    """Placeholder for a random-fill region; the expander replaces it in place and it never reaches
    the printer."""

    def __init__(self, hole: Hole):
        super().__init__("template_hole", is_instrumentation=True)
        self.hole = hole


class Callee:
    """Handle to a declared callee function, returned by `Builder.function` and passed to a call as a
    target. Opaque to templates -- they only hold it and hand it back."""

    __slots__ = ("_function",)

    def __init__(self, function: Function):
        self._function = function


class Builder:
    """The object a template drives: emits into the block of the function being built (the entry, or a
    declared callee's body). Declared callees are shared across every builder of one template."""

    def __init__(self, generator, test_case, function, basic_block, declared=None):
        self._gen = generator
        self._test_case = test_case
        self._function = function
        self._bb = basic_block
        self._declared: List[Function] = declared if declared is not None else []
        self._calls = 0       # calls emitted in this function (capped by max_calls_per_function)
        self._hole_idx = 0     # unique prefix for this function's multi-block holes

    def instruction(self, ref: Union[SpecRef, str], *operands: Union[Reg, Mem, Imm]) -> Instruction:
        """Emit one fixed instruction, selecting the form that matches the given operands."""
        spec = resolve(self._gen.instruction_set, ref, [self._signature(o) for o in operands])
        if spec.control_flow:
            raise ValueError(f"template: instruction() is for straight-line instructions, but {spec.name} "
                             f"is control flow (a block runs start to finish; use call() for calls)")
        inst = spec.generate(self._gen)
        for slot, given in zip(inst.operands, operands):
            self._override(spec.name, slot, given)
        inst.is_from_template = True
        self._warn_if_operand_changed(spec.name, inst, n_pinned=len(operands))
        return self._append(inst)

    def _warn_if_operand_changed(self, iname: str, inst: Instruction, n_pinned: int) -> None:
        """The template is accepted as written -- any instruction is allowed. But if a correctness pass
        would rewrite an operand the template *pinned* (to fix an UNPREDICTABLE register collision), warn
        which operand changes and why. Reuses the real pass to detect the change and to describe the
        constraint (no rule duplication); RNG state is saved/restored so the probe never perturbs seed
        reproducibility. The instruction is left exactly as the template wrote it."""
        patch = next((p for p in self._gen.passes
                      if isinstance(p, Aarch64PatchUndefinedLoadsStoresPass)), None)
        if patch is None:
            return
        rng_state = random.getstate()
        probe = copy.deepcopy(inst)
        patch._patch_instruction(probe)
        random.setstate(rng_state)
        for i in range(min(n_pinned, len(inst.operands), len(probe.operands))):
            if inst.operands[i].value != probe.operands[i].value:
                self._gen.LOG.warning(
                    "template",
                    f"{iname}: your pinned operand '{inst.operands[i].value}' will be rewritten by the "
                    f"correctness pass, because {patch.constraint_reason(inst)}. The instruction is kept "
                    f"as you wrote it; pin non-colliding registers if you want to avoid the rewrite.")

    @staticmethod
    def _signature(given: Union[Reg, Mem, Imm]):
        if isinstance(given, Reg):
            return OT.REG, given.width
        if isinstance(given, Imm):
            return OT.IMM, None
        if isinstance(given, Mem):
            return OT.MEM, None
        raise TypeError(f"template: unsupported operand {given!r}")

    @staticmethod
    def _override(iname: str, slot, given: Union[Reg, Mem, Imm]) -> None:
        if isinstance(given, Reg):
            if slot.type != OT.REG:
                raise ValueError(f"template: {iname} operand is {slot.type.name}, got a register")
            if slot.get_width() != given.width:
                raise ValueError(f"template: {iname} wants a {slot.get_width()}-bit register, "
                                 f"got {given.name}")
            slot.value = given.name.lower()
        elif isinstance(given, Imm):
            if slot.type != OT.IMM:
                raise ValueError(f"template: {iname} operand is {slot.type.name}, got an immediate")
            slot.value = str(given.value)
        elif isinstance(given, Mem):
            if slot.type != OT.MEM:
                raise ValueError(f"template: {iname} operand is {slot.type.name}, got a memory operand")
            base = next((c for c in slot.inner if isinstance(c, RegisterOperand)), None)
            if base is None:
                raise ValueError(f"template: {iname} memory operand has no base register to set")
            base.value = given.base.name.lower()
            # honor an explicit offset (Mem(reg, offset=Imm(n))); else the displacement stays random
            if given.offset is not None:
                off = next((c for c in slot.inner if getattr(c, "mem_role", None) is AArch64MemRole.OFFSET), None)
                if off is None:
                    raise ValueError(f"template: {iname} memory form has no offset slot for the given "
                                     f"offset (this addressing mode takes no displacement)")
                off.value = str(given.offset.value)
            slot.value = ", ".join(c.value for c in slot.inner)
        else:
            raise TypeError(f"template: unsupported operand {given!r}")

    def hole(self, hole: Hole) -> None:
        """Insert a random-fill region. A single-block hole (default) fills the current block; a
        multi-block hole (`n_bb`) lays out that many blocks, wired into a forward DAG (with the same
        conditional/unconditional branches and unreachable flows the random generator emits), and
        continues after them."""
        if resolve_count(hole.n_bb) == 1:
            self._append(HoleInstruction(hole))
            return
        blocks = self._layout_blocks(resolve_count(hole.n_bb))
        counts = [0] * len(blocks)
        for _ in range(resolve_count(hole.n)):
            counts[random.randrange(len(blocks))] += 1
        for blk, c in zip(blocks, counts):
            if c:
                blk.insert_after(blk.get_last(),
                                 HoleInstruction(Hole(c, kind=hole.kind, regs=hole.regs)))

    def _layout_blocks(self, k: int) -> List[BasicBlock]:
        """Splice k blocks (plus unreachable flows) after the current block, wired into a forward DAG by
        the generator, and advance the cursor past them. Returns every spliced block, for filling."""
        header, lo_hi = self._bb, self._gen.successor_bounds()
        prefix = f"{self._function.name.removeprefix('.function_')}_h{self._hole_idx}"
        self._hole_idx += 1
        cont = BasicBlock(f".bb_{prefix}.cont")
        nodes = [BasicBlock(f".bb_{prefix}.{i}") for i in range(k)]
        self._gen._wire_dag(nodes, cont, *lo_hi)
        ordered = self._gen._with_unreachable_flows(prefix, nodes, *lo_hi)
        header.successors = [nodes[0]]
        for blk in ordered + [cont]:
            self._function.append(blk)
        self._bb = cont
        return ordered

    @contextlib.contextmanager
    def if_(self, cond: Optional[Cond] = None):
        """A conditional branch: the block so far ends in one, and the `with` body becomes the taken
        block; control rejoins after it. By default the condition is a random conditional branch (as in
        random test cases); pass a `flags` constant (e.g. `flags.EQ`) to pin it to `b.<cond>`, so the
        input steers taken/not-taken through its NZCV `flags` value."""
        if cond is not None and not isinstance(cond, Cond):
            raise TypeError(f"template: if_(cond=) expects a `flags` constant (e.g. flags.EQ), "
                            f"got {cond!r}")
        header = self._bb
        taken = self._new_bb()
        join = self._new_bb()
        header.successors = [taken, join]
        branch = self._gen.get_conditional_branch_with_condition(taken.name, cond.code) \
            if cond is not None else self._gen.get_conditional_branch_instruction(taken.name)
        header.terminators = [
            branch,
            self._gen.get_unconditional_jump_instruction().add_op(LabelOperand(join.name)),
        ]
        self._bb = taken
        try:
            yield
        finally:
            self._bb.successors = [join]
            self._bb = join

    def function(self, build: Optional[Callable[["Builder"], None]] = None, *,
                 populate: Populate = True) -> Callee:
        """Declare a callee. `build(f)` fills its body via a nested builder; without it the body is a
        random fill per `populate` (True, False, or a Hole). Returns a handle to pass as a call target."""
        func = self._new_function(build, populate)
        self._declared.append(func)
        return Callee(func)

    def call(self, spec: Union[Call, DirectCall, IndirectCall]) -> Instruction:
        """Emit a call to callee(s): `DirectCall` -> `BL`, `IndirectCall` -> `BLR` (materialized here,
        not by the random-path pass), `Call` -> the generator's choice. Targets are declared callees or
        ones created on demand; every target must lie ahead of the caller (forward-only, no loops)."""
        if self._calls >= CONF.max_calls_per_function:
            raise ValueError(f"template: {self._function.name} would exceed max_calls_per_function "
                             f"({CONF.max_calls_per_function})")
        self._calls += 1
        if isinstance(spec, Call):
            spec = DirectCall() if random.random() >= CONF.indirect_call_probability else IndirectCall()
        if isinstance(spec, DirectCall):
            target = self._as_function(spec.target) if spec.target is not None \
                else self._new_function(None, spec.populate)
            self._check_forward(target)
            return self._append(self._gen.get_direct_call_instruction(target.name))
        if isinstance(spec, IndirectCall):
            return self._indirect_call(spec)
        raise TypeError(f"template: call() expects a Call/DirectCall/IndirectCall, got {spec!r}")

    def _indirect_call(self, ic: IndirectCall) -> Instruction:
        callees = self._resolve_targets(ic)
        for c in callees:
            self._check_forward(c)
        reg = INDIRECT_CALL_TARGET_REGISTER
        if self._use_dispatch_table(ic.dispatch, len(callees)):
            # anchor the table on its own first callee so several dispatch calls coexist
            owner = callees[0]
            owner.dispatch_table = DispatchTable(owner, callees)
            index_reg = ic.index_reg.name.lower() if ic.index_reg is not None else None
            for inst in Aarch64IndirectCallPass._dispatch_sequence(reg, owner.dispatch_table, index_reg):
                self._append(inst)
        else:
            self._append(Aarch64IndirectCallPass._adr(reg, callees[0].name))
        return self._append(self._gen.get_indirect_call_instruction(callees[0].name))

    def _resolve_targets(self, ic: IndirectCall) -> List[Function]:
        if isinstance(ic.targets, list):
            return [self._as_function(t) for t in ic.targets]
        n = ic.targets if ic.targets is not None else self._random_target_count()
        # declared callees ahead of the caller come first; create the rest
        pool = [f for f in self._declared if self._index(f) > self._index(self._function)][:n]
        return pool + [self._new_function(None, ic.populate) for _ in range(n - len(pool))]

    @staticmethod
    def _random_target_count() -> int:
        # Unspecified: mirror the random path's single-vs-dispatch choice.
        return random.randint(2, 4) if random.random() < CONF.dispatch_call_probability else 1

    @staticmethod
    def _use_dispatch_table(dispatch, targets: int) -> bool:
        if dispatch is None:
            return targets > 1                      # auto: ADR for one target, a table for several
        if not dispatch and targets > 1:
            raise ValueError("template: IndirectCall(dispatch=False) selects one callee via ADR; it "
                             "cannot choose among several targets -- use dispatch=True for a jump table")
        return dispatch

    def _new_function(self, build, populate: Populate) -> Function:
        """Append a fresh forward function that returns. `build(f)` fills it explicitly, else it is
        populated per `populate` (a random-fill hole, or an empty leaf when False)."""
        idx = len(self._test_case.functions)
        if idx >= CONF.max_functions_per_test_case:
            raise ValueError(f"template: creating another function would exceed "
                             f"max_functions_per_test_case ({CONF.max_functions_per_test_case})")
        func = Function(f".function_{idx}", self._function.owner)
        func.append(BasicBlock(f".bb_{idx}.0"))
        func.exit.terminators = [self._gen.get_return_instruction()]
        self._test_case.functions.append(func)
        sub = Builder(self._gen, self._test_case, func, func.get_first_bb(), self._declared)
        if build is not None:
            build(sub)
        elif populate is not False:
            sub.hole(populate if isinstance(populate, Hole) else Hole(n=None))
        sub._finalize()
        return func

    def _new_bb(self) -> BasicBlock:
        suffix = self._function.name.removeprefix(".function_")
        bb = BasicBlock(f".bb_{suffix}.{len(self._function)}")
        self._function.append(bb)
        return bb

    def _finalize(self) -> None:
        if not self._bb.successors:
            self._bb.successors = [self._function.exit]

    def _as_function(self, target) -> Function:
        if not isinstance(target, Callee):
            raise TypeError(f"template: a call target must be a callee from Builder.function(), "
                            f"got {target!r}")
        return target._function

    def _index(self, func: Function) -> int:
        return self._test_case.functions.index(func)

    def _check_forward(self, target: Function) -> None:
        if self._index(target) <= self._index(self._function):
            raise ValueError(f"template: {self._function.name} cannot call {target.name}: a call target "
                             f"must lie ahead of the caller (forward-only call graph, no loops)")

    def _append(self, inst: Instruction) -> Instruction:
        self._bb.insert_after(self._bb.get_last(), inst)
        return inst


class Template(abc.ABC):
    """A template: subclass and implement `build`, emitting fixed ops, holes, and gadgets on `b`."""

    @abc.abstractmethod
    def build(self, b: Builder) -> None:
        ...
