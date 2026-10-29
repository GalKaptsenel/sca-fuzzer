"""GlobalSealTable (GST): an in-register table of per-input seal values.

Like a GOT but in a register: a seal reads its value from the table (loaded from the input).
"""
from abc import ABC, abstractmethod
from typing import List

from ...interfaces import Instruction, RegisterOperand, SIMD_SUBREGION_SIZE
from ..reif import REIFSectionType

_UMOV_LANE = {1: "b", 2: "h", 4: "s", 8: "d"}       # read width (bytes) -> UMOV lane
VECTOR_REG_BYTES = 16                                # one 128-bit register
SIMD_BASE_REG = 16                                   # SIMD image slot j -> v(SIMD_BASE_REG + j)
TOP_REG = 31


def _lane(width: int) -> int:
    for lane in (1, 2, 4, 8):
        if width <= lane:
            return lane
    raise ValueError(f"value of {width} bytes exceeds the widest single read (8)")


class Location(ABC):
    """Where one value is stored; the holder never sees a register name or offset."""

    @abstractmethod
    def load(self, tmp_reg: str) -> List[Instruction]:
        """Instructions reading the stored value into tmp_reg."""

    @abstractmethod
    def byte_range(self) -> range:
        """The value's byte offsets within this file's REIF section."""


class Image(ABC):
    """A fresh per-input store for the table; filled, then serialized to its REIF section."""

    @abstractmethod
    def store(self, location: Location, value: int) -> None:
        """Write `value` little-endian (truncated to the location's width) at `location`."""

    @abstractmethod
    def to_bytes(self) -> bytes:
        """This image as the REIF section payload."""


class GlobalSealTable(ABC):
    """Register storage for seal values; the backing registers (vector/GPR) and count are hidden.
    Built with a bit capacity; raises GlobalSealTableCapacityExceeded when allocation exceeds it."""

    @abstractmethod
    def allocate(self, width: int) -> Location:
        """Reserve `width` bytes; raise on overflow."""

    @abstractmethod
    def blank_image(self) -> Image:
        """A fresh, zeroed per-input store for this file."""

    @property
    @abstractmethod
    def section(self) -> REIFSectionType:
        """Which REIF section an image of this file fills."""


class GlobalSealTableCapacityExceeded(Exception):
    """Allocation exceeded the table's capacity."""


class _VectorLocation(Location):
    def __init__(self, reg: int, byte_offset: int, width: int) -> None:
        self._reg = reg
        self._byte_offset = byte_offset
        self._width = width

    def load(self, tmp_reg: str) -> List[Instruction]:
        lane = _UMOV_LANE[self._width]
        idx = self._byte_offset // self._width
        dst = tmp_reg if 8 == self._width else "w" + tmp_reg[1:]   # UMOV writes W for < 64-bit
        return [Instruction("umov", True, "", False,
                            template=f"UMOV {dst}, v{self._reg}.{lane}[{idx}]")
                .add_op(RegisterOperand(dst, self._width * 8, False, True))
                .add_op(RegisterOperand(f"v{self._reg}", 128, True, False))]

    def byte_range(self) -> range:
        base = (self._reg - SIMD_BASE_REG) * VECTOR_REG_BYTES + self._byte_offset
        return range(base, base + self._width)


class _VectorImage(Image):
    def __init__(self) -> None:
        self._bytes = bytearray(SIMD_SUBREGION_SIZE)

    def store(self, location: Location, value: int) -> None:
        rng = location.byte_range()
        width = len(rng)
        masked = value & ((1 << (8 * width)) - 1)
        self._bytes[rng.start:rng.stop] = masked.to_bytes(width, "little")

    def to_bytes(self) -> bytes:
        return bytes(self._bytes)


class VectorGlobalSealTable(GlobalSealTable):
    """Seal-value storage backed by the high vector registers (v31 down). `capacity_bits` is rounded
    up to whole registers; values are byte-aligned, top register first, never straddling one."""

    def __init__(self, capacity_bits: int) -> None:
        if capacity_bits <= 0:
            raise ValueError("capacity_bits must be positive")
        n_regs = -(-capacity_bits // (VECTOR_REG_BYTES * 8))   # ceil to whole registers
        if n_regs > SIMD_SUBREGION_SIZE // VECTOR_REG_BYTES:
            raise ValueError(f"capacity_bits {capacity_bits} exceeds the SIMD section")
        self._lowest_reg = TOP_REG + 1 - n_regs
        self._reg = TOP_REG
        self._byte = 0

    def allocate(self, width: int) -> Location:
        lane = _lane(width)
        start = self._byte + (-self._byte % lane)              # align up within the register
        if start + lane > VECTOR_REG_BYTES:                    # does not fit -> next register down
            self._reg -= 1
            start = 0
        if self._reg < self._lowest_reg:
            raise GlobalSealTableCapacityExceeded(
                f"capacity exhausted (v{self._lowest_reg}..v{TOP_REG}); cannot place {width} bytes")
        self._byte = start + lane
        return _VectorLocation(self._reg, start, lane)

    def blank_image(self) -> Image:
        return _VectorImage()

    @property
    def section(self) -> REIFSectionType:
        return REIFSectionType.SIMD
