"""Refer to an instruction by name and resolve it against the loaded InstructionSet at build time.

A template names an instruction as `specs.LDR` (a `SpecRef`); resolution, and the not-found error,
happen when the builder emits it -- so no spec objects need to exist at import time.
"""
from __future__ import annotations

from typing import List, Optional, Tuple, Union

from ...interfaces import InstructionSpec, InstructionSetAbstract as InstructionSet, OT

Signature = List[Tuple[OT, Optional[int]]]     # per-operand (type, width) requested by the caller


class SpecRef:
    """A by-name reference to an instruction, resolved against the loaded set at build time."""
    def __init__(self, name: str):
        self.name = name

    def __repr__(self) -> str:
        return f"SpecRef({self.name!r})"


def _fits(spec: InstructionSpec, want: Signature) -> bool:
    if len(spec.operands) != len(want):
        return False
    return all(o.type == t and (w is None or o.width == w) for o, (t, w) in zip(spec.operands, want))


def resolve(instruction_set: InstructionSet, ref: Union[SpecRef, str],
            want: Optional[Signature] = None) -> InstructionSpec:
    """Find the named instruction. With `want` (the caller's operand signature) the matching form is
    selected -- an instruction may have several (e.g. 32/64-bit, immediate/register-offset LDR)."""
    name = ref.name if isinstance(ref, SpecRef) else ref
    matches = [s for s in instruction_set.instructions if s.name.lower() == name.lower()]
    if not matches:
        raise ValueError(f"template: no instruction '{name}' in the loaded set")
    if want is None:
        return matches[0]
    fit = [s for s in matches if _fits(s, want)]
    if fit:
        return fit[0]
    forms = [[(o.type.name, o.width) for o in s.operands] for s in matches]
    raise ValueError(f"template: no form of '{name}' takes operands {want}; available: {forms}")


class _SpecTable:
    """Attribute access -> SpecRef, so `specs.LDR` is `SpecRef('LDR')` regardless of what is loaded;
    resolution (and the not-found error) happens when the builder emits it."""
    def __getattr__(self, name: str) -> SpecRef:
        return SpecRef(name)


specs = _SpecTable()
