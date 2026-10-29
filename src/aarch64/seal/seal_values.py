"""The per-input values a seal reads from the GST, as an interface + one concrete per value.

A SealValue knows its width (bytes in the register) and how the fixed code applies it to a target
register after it has been loaded into a scratch GPR. The numeric value itself is written into the
per-input Image by the resolver; a SealValue only describes the type's shape and apply op.
"""
from abc import ABC, abstractmethod
from typing import List

from ...interfaces import Instruction, RegisterOperand, ImmediateOperand
from .global_seal_table import Location


class SealValue(ABC):
    @property
    @abstractmethod
    def width(self) -> int:
        """Bytes this value occupies in its register."""

    @abstractmethod
    def apply(self, location: Location, target_reg: str, tmp_reg: str) -> List[Instruction]:
        """Read the value from `location` and apply it to `target_reg` (tmp_reg as scratch)."""


class PacSignature(SealValue):
    """A PAC signature inserted into the pointer's PAC field [lsb, lsb+nbits)."""

    def __init__(self, lsb: int, nbits: int) -> None:
        self._lsb = lsb
        self._nbits = nbits

    @property
    def width(self) -> int:
        return -(-self._nbits // 8)

    def apply(self, location: Location, target_reg: str, tmp_reg: str) -> List[Instruction]:
        return location.load(tmp_reg) + [_bfi(target_reg, tmp_reg, self._lsb, self._nbits)]


class MteTag(SealValue):
    """A 4-bit allocation tag inserted into the pointer's tag field (bits [59:56])."""

    width = 1

    def apply(self, location: Location, target_reg: str, tmp_reg: str) -> List[Instruction]:
        return location.load(tmp_reg) + [_bfi(target_reg, tmp_reg, 56, 4)]


class MaskFlip(SealValue):
    """A 64-bit mask XORed into a register (canonicality / branch-target / sign-extension flips)."""

    width = 8

    def apply(self, location: Location, target_reg: str, tmp_reg: str) -> List[Instruction]:
        return location.load(tmp_reg) + [_eor(target_reg, tmp_reg)]


def _bfi(target_reg: str, tmp_reg: str, lsb: int, nbits: int) -> Instruction:
    return (Instruction("bfi", True, "", False,
                        template=f"BFI {target_reg}, {tmp_reg}, #{lsb}, #{nbits}")
            .add_op(RegisterOperand(target_reg, 64, True, True))
            .add_op(RegisterOperand(tmp_reg, 64, True, False))
            .add_op(ImmediateOperand(str(lsb), 6))
            .add_op(ImmediateOperand(str(nbits), 6)))


def _eor(target_reg: str, tmp_reg: str) -> Instruction:
    return (Instruction("eor", True, "", False,
                        template=f"EOR {target_reg}, {target_reg}, {tmp_reg}")
            .add_op(RegisterOperand(target_reg, 64, False, True))
            .add_op(RegisterOperand(target_reg, 64, True, False))
            .add_op(RegisterOperand(tmp_reg, 64, True, False)))
