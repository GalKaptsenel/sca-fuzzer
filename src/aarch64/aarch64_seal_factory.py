"""Factories that wire the decoupled seals into a SealInstrumentation pass.

Each primitive contributes one pure value-seal (composed on top of the Sandbox clamp, in order) and
the FixPoint subclass that holds its per-input data. The caller passes the set of active primitives;
which one is the non-interference target (decoyed) is the executor's decoy policy, not the pass's.
"""
from typing import Set

from .aarch64_generator import Aarch64Generator
from .aarch64_mte import SealInstrumentation, MteTag, MTEFixPoint


def make_seal_pass(generator: Aarch64Generator, primitives: Set[str]) -> SealInstrumentation:
    value_seals = []
    fixpoint_cls = MTEFixPoint
    if "mte" in primitives:
        value_seals.append(MteTag())
        fixpoint_cls = MTEFixPoint
    if not value_seals:
        raise ValueError(f"no sealable primitives in {primitives!r}")
    return SealInstrumentation(generator, value_seals, fixpoint_cls)
