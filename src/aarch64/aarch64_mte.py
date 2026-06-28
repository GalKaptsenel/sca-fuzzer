"""MTE allocation-tag modelling used while parsing the contract-executor trace.

MteTagState tracks the allocation tag of each memory granule across speculation. It is a stack of
layers indexed by speculation depth: layer 0 is architectural; entering deeper speculation copies
the current top, and unwinding pops it — that pop is the revert of the speculative tag stores. It
starts uniform (the region's initial tag); per-cell initial tags can be pre-seeded for dynamic
tagging.
"""
from typing import Dict, List, Optional, Tuple

from .aarch64_disasm import decode_tag_store

MTE_GRANULE = 16  # bytes covered by one allocation tag
MTE_INITIAL_TAG = 6  # uniform allocation tag the sandbox is loaded with (kernel + CE + model agree)

# STG-family memory-tag stores -> number of granules each tags.
_MTE_TAG_STORES = {"stg": 1, "stzg": 1, "st2g": 2, "stz2g": 2}


class MteTagState:
    def __init__(self, default_tag: int):
        self._default = default_tag & 0xF
        self._stack: List[Dict[int, int]] = [{}]  # one tag-override layer per live speculation depth

    @staticmethod
    def granule(addr: int) -> int:
        return (addr & ((1 << 56) - 1)) & ~(MTE_GRANULE - 1)  # drop the tag byte and in-granule offset

    def to_depth(self, nesting: int) -> None:
        """Track the current speculation depth: grow by copying the top (speculation inherits the
        live state), shrink by popping (reverting the deeper levels' speculative stores)."""
        while len(self._stack) <= nesting:
            self._stack.append(dict(self._stack[-1]))
        del self._stack[nesting + 1:]

    def set(self, addr: int, tag: int, n_granules: int = 1) -> None:
        """Tag granules in the current (deepest live) layer."""
        layer, g = self._stack[-1], self.granule(addr)
        for i in range(n_granules):
            layer[g + i * MTE_GRANULE] = tag & 0xF

    def tag_at(self, addr: int) -> int:
        """The tag visible at the current speculation depth (architectural + live speculative stores)."""
        return self._stack[-1].get(self.granule(addr), self._default)


def _reg_value(cpu, name: str) -> int:
    name = name.lower()
    if name == "sp":
        return cpu.sp
    if name.startswith("x") and name[1:].isdigit():
        return cpu.gpr[int(name[1:])]
    return 0


def mte_tag_store_effect(ite) -> Optional[Tuple[int, int, int]]:
    """If ite is an STG-family tag store, return (addr, tag, n_granules); else None. STG writes the
    allocation TAG of the granule at the base register's address (not data memory) — the CE flags no
    memory access and exposes no effective address, so the granule address (base + disp) and the tag
    source register come from capstone's structured operands. The tag is the logical tag of Xt."""
    dec = decode_tag_store(ite.cpu.encoding, ite.cpu.pc)
    if dec is None:
        return None
    mn, xt, base, disp = dec
    return _reg_value(ite.cpu, base) + disp, (_reg_value(ite.cpu, xt) >> 56) & 0xF, _MTE_TAG_STORES[mn]


