"""Factories that wire the decoupled seals into a SealInstrumentation memory pass.

Each active primitive contributes one pure value-seal (composed on top of the Sandbox clamp by the
pass, in list order) and its FixPoint data. Only Sandbox clamps; PacSign/MteTag never sandbox. PAC's
other half — sealing generator-emitted AUT* instructions — is a separate pass (PacAuthInstrumentation),
not a memory value-seal.
"""
from dataclasses import dataclass
from typing import Set

from .aarch64_generator import Aarch64Generator
from .aarch64_seal import SealInstrumentation
from .aarch64_mte import MteTag, MTEFixPoint
from .aarch64_pac import PacSign, PACFixPoint, build_pac_specs


@dataclass
class PacMteFixPoint(PACFixPoint, MTEFixPoint):
    """A memory access composed with both PacSign and MteTag — carries both seals' data. reset()
    chains PACFixPoint -> MTEFixPoint -> FixPoint via the MRO, clearing every field."""


_FIXPOINT_CLS = {
    frozenset({"pac"}): PACFixPoint,
    frozenset({"mte"}): MTEFixPoint,
    frozenset({"pac", "mte"}): PacMteFixPoint,
}


def make_seal_pass(generator: Aarch64Generator, primitives: Set[str]) -> SealInstrumentation:
    """A memory-sealing pass composing [Sandbox] + the active primitives' value-seals (PacSign before
    MteTag): pac -> [Sandbox, PacSign], mte -> [Sandbox, MteTag], both -> [Sandbox, PacSign, MteTag]."""
    value_seals = []
    if "pac" in primitives:
        _, auth_specs, xpac_specs = build_pac_specs(generator)
        value_seals.append(PacSign(generator, auth_specs, xpac_specs))
    if "mte" in primitives:
        value_seals.append(MteTag())
    fixpoint_cls = _FIXPOINT_CLS.get(frozenset(primitives))
    if fixpoint_cls is None:
        raise ValueError(f"unsupported memory-seal primitives: {primitives!r}")
    return SealInstrumentation(generator, value_seals, fixpoint_cls)
