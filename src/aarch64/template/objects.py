"""Typed vocabulary for templates: registers, memory operands, immediates, holes, and calls -- so a
template is built from objects, validated at build time, with no assembly-string parsing."""
from __future__ import annotations

from dataclasses import dataclass, KW_ONLY
from enum import Enum
from typing import List, Optional, Tuple, Union


class Kind(Enum):
    """The class of instruction a hole is filled with -- all non-control-flow, so the block still runs
    start to finish."""
    ANY = "any"
    LOAD = "load"
    STORE = "store"
    ALU = "alu"    # any non-memory, non-control-flow instruction


@dataclass(frozen=True)
class Reg:
    """A GPR operand. `number` is the register index (X0 and W0 share n=0); None for sp/zr."""
    name: str
    width: int
    number: Optional[int] = None

    def __str__(self) -> str:
        return self.name


# Importable register constants: X0..X30 / W0..W30 plus the special names.
_g = globals()
for _i in range(31):
    _g[f"X{_i}"] = Reg(f"x{_i}", 64, _i)
    _g[f"W{_i}"] = Reg(f"w{_i}", 32, _i)
SP = Reg("sp", 64)
XZR = Reg("xzr", 64)
WZR = Reg("wzr", 32)


@dataclass(frozen=True)
class Imm:
    """An immediate operand. `value` is required; `width` is optional."""
    value: int
    _: KW_ONLY
    width: int = 64

    def __post_init__(self):
        if not isinstance(self.value, int):
            raise TypeError("Imm: value must be an int")


@dataclass(frozen=True)
class Mem:
    """A memory operand. `base` register is required; `offset`/`width` are optional."""
    base: Reg
    _: KW_ONLY
    offset: Optional[Imm] = None
    width: int = 64

    def __post_init__(self):
        if not isinstance(self.base, Reg):
            raise TypeError("Mem: base must be a register (e.g. X1)")
        if self.offset is not None and not isinstance(self.offset, Imm):
            raise TypeError("Mem: offset must be an Imm")


Count = Union[int, Tuple[int, int], None]
"""How many instructions a hole holds: an exact int, a (min, max) range, or None for a random count."""


def _validate_count(n: Count) -> None:
    if n is None:
        return
    if isinstance(n, bool):
        raise TypeError("Hole: n must be an int, a (min, max) tuple, or None")
    if isinstance(n, int):
        if n < 1:
            raise ValueError("Hole: n must be >= 1")
        return
    if isinstance(n, tuple) and len(n) == 2 and all(isinstance(x, int) for x in n):
        lo, hi = n
        if not 1 <= lo <= hi:
            raise ValueError("Hole: n range must be (min, max) with 1 <= min <= max")
        return
    raise TypeError("Hole: n must be an int, a (min, max) tuple, or None")


@dataclass
class Hole:
    """A random-fill region. `n` is how many instructions and `n_bb` how many basic blocks (each a Count:
    an exact int, a (min, max) range, or None for random; default one each). Over several blocks the
    generator wires conditional branches between them, so each block still runs start to finish. `kind`
    narrows the instruction class and `regs` the registers -- unset, each is randomized."""
    n: Count = 1
    _: KW_ONLY
    n_bb: Count = 1
    kind: Kind = Kind.ANY
    regs: Optional[List[Reg]] = None

    def __post_init__(self):
        _validate_count(self.n)
        _validate_count(self.n_bb)
        if not isinstance(self.kind, Kind):
            raise TypeError("Hole: kind must be a Kind")
        if self.regs is not None and not all(isinstance(r, Reg) for r in self.regs):
            raise TypeError("Hole: regs must be a list of registers")


Populate = Union[bool, "Hole"]
"""A created callee's body: True for a random body, False for an empty leaf, or a Hole to constrain it
(same count/kind/regs options as any hole)."""


def _validate_populate(p: Populate) -> None:
    if not isinstance(p, (bool, Hole)):
        raise TypeError("populate must be True, False, or a Hole")


@dataclass
class DirectCall:
    """A direct call (`BL`). `target` is a declared callee (from `Builder.function`); unset, one is
    created and given a body per `populate` (see Populate)."""
    _: KW_ONLY
    target: Optional[object] = None    # a Callee handle
    populate: Populate = True

    def __post_init__(self):
        _validate_populate(self.populate)


@dataclass
class IndirectCall:
    """An indirect call (`BLR`). `targets` is the callee set: an int count (declared callees are used
    first, the rest created), a list of declared callees, or None (generator's choice). `dispatch` picks
    the mechanism: None auto (`ADR` for one, a jump table for several), True a jump table (may hold one
    entry), False a direct `ADR` (one target only). `populate` (see Populate) bodies created callees."""
    _: KW_ONLY
    targets: Optional[Union[int, list]] = None
    dispatch: Optional[bool] = None
    index_reg: Optional[Reg] = None
    populate: Populate = True

    def __post_init__(self):
        if isinstance(self.targets, bool):
            raise TypeError("IndirectCall: targets must be an int count, a list of callees, or None")
        if isinstance(self.targets, int) and self.targets < 1:
            raise ValueError("IndirectCall: targets count must be >= 1")
        if isinstance(self.targets, list) and not self.targets:
            raise ValueError("IndirectCall: targets list must not be empty")
        if self.dispatch is not None and not isinstance(self.dispatch, bool):
            raise TypeError("IndirectCall: dispatch must be a bool")
        if self.index_reg is not None and not isinstance(self.index_reg, Reg):
            raise TypeError("IndirectCall: index_reg must be a register (e.g. X2)")
        _validate_populate(self.populate)


@dataclass
class Call:
    """A function call with the kind left to the generator -- direct (`BL`) or indirect (`BLR`), chosen
    per the configured indirect-call probability. Use DirectCall or IndirectCall to pin the kind."""


@dataclass(frozen=True)
class Cond:
    """An AArch64 condition code, tested against NZCV by a flag-conditioned branch (b.<code>). Use the
    `flags` constants (flags.EQ, flags.NE, ...) rather than constructing this directly."""
    code: str


class _Flags:
    """The AArch64 condition codes, for pinning a conditional branch: `b.if_(flags.EQ)`. The branch is
    then taken per NZCV, which the input controls via its `flags` register value."""
    EQ = Cond("eq"); NE = Cond("ne")
    CS = Cond("cs"); HS = Cond("cs"); CC = Cond("cc"); LO = Cond("cc")
    MI = Cond("mi"); PL = Cond("pl"); VS = Cond("vs"); VC = Cond("vc")
    HI = Cond("hi"); LS = Cond("ls"); GE = Cond("ge"); LT = Cond("lt")
    GT = Cond("gt"); LE = Cond("le")


flags = _Flags()


__all__ = (
    ["Kind", "Reg", "Imm", "Mem", "Hole", "Call", "DirectCall", "IndirectCall", "SP", "XZR", "WZR",
     "Cond", "flags"]
    + [f"X{i}" for i in range(31)]
    + [f"W{i}" for i in range(31)]
)
