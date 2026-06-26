"""MTE allocation-tag modelling used while parsing the contract-executor trace.

MteTagState tracks the allocation tag of each memory granule across speculation. It is a stack of
layers indexed by speculation depth: layer 0 is architectural; entering deeper speculation copies
the current top, and unwinding pops it — that pop is the revert of the speculative tag stores. It
starts uniform (the region's initial tag); per-cell initial tags can be pre-seeded for dynamic
tagging.
"""
from typing import Dict, List, Optional, Tuple
import random
from dataclasses import dataclass

from ..interfaces import Instruction
from .aarch64_disasm import decode_tag_store
from .aarch64_seal import Seal, FixPoint, make_nop

MTE_GRANULE = 16  # bytes covered by one allocation tag

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


# ===========================================================================
# MTE non-interference seal + sealing pass
# ===========================================================================

MTE_SLOT_SIZE = 1  # one slot instruction per memory access

class MteTag(Seal):
    """MTE allocation-tag seal.  Slot = a single instruction.

    Genuine is a NOP — the sandboxed pointer already carries the region's (correct) tag. A decoy
    retags it: a hardware-random tag (IRG) or a hardcoded one (EOR a tag-field mask). Both touch
    only the tag bits [59:56], leaving the address unchanged, and assume no fixed region tag.
    """
    name = "mte_tag"
    slot_size = MTE_SLOT_SIZE

    # EOR masks confined to the tag field [59:56] (each a valid AArch64 logical immediate); a
    # nonzero one yields a different tag while leaving the address bits untouched.
    _TAG_FLIP_MASKS = (0x1 << 56, 0x2 << 56, 0x4 << 56, 0x8 << 56, 0xF << 56)

    def _irg(self, reg: str) -> Instruction:
        return Instruction("irg", True, "", False, template=f"IRG {reg}, {reg}")

    def _retag(self, reg: str, mask: int) -> Instruction:
        return Instruction("eor", True, "", False, template=f"EOR {reg}, {reg}, #0x{mask:016x}")

    def _addg(self, reg: str, tag_delta: int) -> Instruction:
        # add a 4-bit offset to the pointer's tag, address offset 0 (a clean mod-16 add when
        # GCR_EL1.Exclude == 0); touches only the tag field.
        return Instruction("addg", True, "", False, template=f"ADDG {reg}, {reg}, #0, #{tag_delta}")

    # ---- Seal protocol (stateless; everything it needs is on the fix point) ----
    def placeholder(self, fp) -> List[Instruction]:
        return [make_nop()]

    def genuine(self, fp, rng: random.Random) -> List[Instruction]:
        # The arch path must carry the cell's tag; fix the pointer's tag only when it mismatches.
        if fp.correct_tag is None or fp.ptr_tag is None:
            return [make_nop()]
        delta = (fp.correct_tag - fp.ptr_tag) % 16
        return [make_nop()] if delta == 0 else [self._addg(fp.value_reg, delta)]

    def decoy(self, fp, rng: random.Random) -> List[Instruction]:
        if rng.random() < 0.5:
            return [self._irg(fp.value_reg)]
        return [self._retag(fp.value_reg, rng.choice(self._TAG_FLIP_MASKS))]


@dataclass
class MTEFixPoint(FixPoint):
    # Per-input, from the sealing trace (reset between inputs):
    correct_tag: Optional[int] = None  # the accessed cell's allocation tag (from MteTagState)
    ptr_tag: Optional[int] = None       # the tag the pointer itself carries (top byte of the EA)

    def reset(self) -> None:
        super().reset()
        self.correct_tag = None
        self.ptr_tag = None

    def resolve(self, cer, layout) -> None:
        """Classify the accessed cell's tag and the slot's min speculation depth. The guarded access
        is self.trigger; the region tag comes from x29 (the sandbox base) in the trace."""
        if cer and self.trigger is not None:
            access_off, code_base = layout.instruction_address[self.trigger], cer[0].cpu.pc
            tags = MteTagState((cer[0].cpu.gpr[29] >> 56) & 0xF)
            for ite in cer:
                nest = ite.metadata.speculation_nesting
                tags.to_depth(nest)
                store = mte_tag_store_effect(ite)
                if store is not None:
                    tags.set(*store)
                if not ite.metadata.has_memory_access or ite.cpu.pc - code_base != access_off:
                    continue
                ea = ite.metadata.memory_access.effective_address
                if self.spec_nesting is None or nest < self.spec_nesting:
                    self.spec_nesting = int(nest)
                if nest == 0 or self.correct_tag is None:   # architectural occurrence is authoritative
                    self.correct_tag, self.ptr_tag = tags.tag_at(ea), (ea >> 56) & 0xF
        super().resolve(cer, layout)
