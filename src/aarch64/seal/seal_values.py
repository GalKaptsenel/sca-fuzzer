"""The per-input values a seal reads from the GVT, as an interface + one concrete per value.

A SealValue owns one field's shape: its width (the bytes it occupies in the register, read with the
smallest UMOV element that covers it), how the per-input numeric value is encoded into those bytes
(right-aligned), and how the fixed code applies it to a target register after loading it into a scratch
GPR. Storing each field right-aligned in its minimal lane lets many values pack into few vector
registers. The numeric value itself is written into the per-input Image by the resolver (as encode()).
"""
from abc import ABC, abstractmethod
from typing import List

from ...interfaces import Instruction, RegisterOperand, ImmediateOperand
from .global_value_table import Location


def _bytes_for(nbits: int) -> int:
    return -(-nbits // 8)


class SealValue(ABC):
    @property
    @abstractmethod
    def width(self) -> int:
        """Bytes this value occupies in its register (the UMOV element covering its field)."""

    @abstractmethod
    def encode(self, value: int) -> int:
        """The numeric value as stored in the lane: the field's bits, right-aligned to bit 0."""

    @abstractmethod
    def apply(self, location: Location, target_reg: str, tmp_reg: str) -> List[Instruction]:
        """Read the value from `location` into tmp_reg and apply it to `target_reg`."""


class PacSignature(SealValue):
    """A PAC signature: the pointer's PAC-field bits occupy the window [lsb, lsb+nbits). The window is
    the bounding box of the field's runs — any gap bit it covers (e.g. bit 55) holds the pointer's own
    canonical value, so re-inserting it is idempotent. The lane holds the window right-aligned; one BFI
    puts it back at [lsb, lsb+nbits)."""

    def __init__(self, lsb: int, nbits: int) -> None:
        self._lsb = lsb
        self._nbits = nbits

    @property
    def lsb(self) -> int:
        return self._lsb

    @property
    def width(self) -> int:
        return _bytes_for(self._nbits)

    def encode(self, value: int) -> int:
        return value >> self._lsb

    def apply(self, location: Location, target_reg: str, tmp_reg: str) -> List[Instruction]:
        return location.load(tmp_reg) + [_bfi(target_reg, tmp_reg, self._lsb, self._nbits)]


class MteTag(SealValue):
    """A 4-bit allocation tag inserted into the pointer's tag field [56, 60). The tag is already
    right-aligned, so the lane is one byte; one BFI puts it at [56, 60)."""

    width = 1

    def encode(self, value: int) -> int:
        return value & 0xF

    def apply(self, location: Location, target_reg: str, tmp_reg: str) -> List[Instruction]:
        return location.load(tmp_reg) + [_bfi(target_reg, tmp_reg, 56, 4)]


class MaskFlip(SealValue):
    """A bitmask XORed into a register (canonicality / branch-target / sign-extension flips). The mask's
    set bits lie in the window [lo, lo+nbits); the lane holds it right-aligned, and one shifted EOR
    (`EOR target, target, tmp, LSL #lo`) puts it back at its window and flips those bits."""

    def __init__(self, lo: int, nbits: int) -> None:
        self._lo = lo
        self._nbits = nbits

    @property
    def width(self) -> int:
        return _bytes_for(self._nbits)

    def encode(self, value: int) -> int:
        return value >> self._lo

    def apply(self, location: Location, target_reg: str, tmp_reg: str) -> List[Instruction]:
        return location.load(tmp_reg) + [_eor(target_reg, tmp_reg, self._lo)]


def _bfi(target_reg: str, tmp_reg: str, lsb: int, nbits: int) -> Instruction:
    return (Instruction("bfi", True, "", False,
                        template=f"BFI {target_reg}, {tmp_reg}, #{lsb}, #{nbits}")
            .add_op(RegisterOperand(target_reg, 64, True, True))
            .add_op(RegisterOperand(tmp_reg, 64, True, False))
            .add_op(ImmediateOperand(str(lsb), 6))
            .add_op(ImmediateOperand(str(nbits), 6)))


def _eor(target_reg: str, tmp_reg: str, lo: int) -> Instruction:
    shift = f", lsl #{lo}" if lo else ""
    return (Instruction("eor", True, "", False,
                        template=f"EOR {target_reg}, {target_reg}, {tmp_reg}{shift}")
            .add_op(RegisterOperand(target_reg, 64, False, True))
            .add_op(RegisterOperand(target_reg, 64, True, False))
            .add_op(RegisterOperand(tmp_reg, 64, True, False)))
