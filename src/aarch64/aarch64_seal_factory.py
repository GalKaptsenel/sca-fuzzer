"""Factories that wire the decoupled seals into a SealInstrumentation memory pass.

Each primitive contributes one pure value-seal (composed on top of the Sandbox clamp by the pass)
and the FixPoint subclass holding its per-input data. Only Sandbox clamps; PacSign/MteTag never
sandbox. PAC's other half — sealing the generator's AUT* instructions — is a separate pass
(PacAuthInstrumentation), not a memory value-seal.
"""
from typing import Set

from .aarch64_generator import Aarch64Generator
from .aarch64_mte import SealInstrumentation, MteTag, MTEFixPoint
from .aarch64_pac import PacSign, PACFixPoint, build_pac_specs


def make_seal_pass(generator: Aarch64Generator, primitives: Set[str]) -> SealInstrumentation:
    """A memory-sealing pass for one primitive (the non-interference target that owns memory):
    pac -> [Sandbox, PacSign], mte -> [Sandbox, MteTag]."""
    if primitives == {"mte"}:
        return SealInstrumentation(generator, [MteTag()], MTEFixPoint)
    if primitives == {"pac"}:
        _, auth_specs, xpac_specs = build_pac_specs(generator)
        return SealInstrumentation(generator, [PacSign(generator, auth_specs, xpac_specs)], PACFixPoint)
    raise NotImplementedError(f"unsupported memory-seal primitives: {primitives!r}")
