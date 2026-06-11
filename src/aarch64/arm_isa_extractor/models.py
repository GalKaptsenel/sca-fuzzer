from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum


class ExtractionError(Exception):
    pass


class OperandKind(Enum):
    REG = "reg"
    IMM = "imm"
    COND = "cond"
    EXTEND = "extend"   # shift/extend type: LSL/LSR/ASR/ROR/UXTB..SXTX (shifted/extended reg or address)
    LABEL = "label"
    FLAGS = "flags"


class MemRole(Enum):
    NONE = "none"      # not part of a memory address
    BASE = "base"      # address base register inside `[...]` (the `<Xn|SP>`)
    INDEX = "index"    # address index register inside `[...]` (the `<Xm>`)
    OFFSET = "offset"  # numeric address immediate inside `[...]` (displacement or index shift amount)
    EXTEND = "extend"  # address index shift/extend type inside `[...]` (LSL/ASR/UXTW/SXTW/...)


class MemAccess(Enum):
    NONE = "none"
    LOAD = "load"
    STORE = "store"
    RMW = "rmw"
    EX_LOAD = "ex-load"
    EX_STORE = "ex-store"
    PREFETCH = "prefetch"


class RegFile(Enum):
    GP = "gp"
    SIMD = "simd"      # B/H/S/D/Q scalar and V vector (FP/AdvSIMD)
    SVE_Z = "sve_z"
    SVE_P = "sve_p"


@dataclass(frozen=True)
class FlagEffects:
    written: frozenset  # subset of {"N","Z","C","V"}
    read: frozenset


@dataclass(frozen=True)
class EncodingCtx:
    """Everything needed to resolve one encoding's operand values."""
    boxes: dict           # field name -> bit width
    const_fields: dict    # fields with a fixed value in this encoding (for scale resolution)
    arch_constants: dict  # architecture-wide ASL integer constants (e.g. LOG2_TAG_GRANULE)
    decode: str           # the Decode ASL text
    execute: str          # the Execute ASL text (e.g. for PC-relative label detection)


@dataclass(frozen=True)
class Operand:
    name: str
    kind: OperandKind
    read: bool
    write: bool
    width: int
    signed: bool
    values: tuple = ()                 # enumerated symbols (COND/shift), else empty
    # IMM value set as (esize, lo, hi, stride) entries. esize 0 = applies for any element size
    # (the usual case, one entry); for SVE tsz-encoded shifts the valid range depends on the
    # element size, so there is one entry per esize (8/16/32/64).
    imm_ranges: tuple = ()
    reg_file: "RegFile | None" = None  # for register operands
    asl_index: str | None = None
    sp_capable: bool = False           # register 31 means SP (else XZR)
    # the encodable register numbers as (lo, hi, stride). Not every instruction can use all 32:
    # a narrow field (Pg is 3-bit => P0-P7) or fixed bits in the UInt() binding (Rm::'0' => even
    # regs) restrict the set. () only for non-register operands.
    reg_range: tuple = ()
    # a register inside a memory `[...]` is the address base or index; the access itself is the
    # instruction's mem_access. NONE for ordinary (non-addressing) registers.
    mem_role: MemRole = MemRole.NONE


@dataclass(frozen=True)
class Instruction:
    name: str
    iclass_id: str
    category: str
    encoding_name: str
    asm_template: str
    control_flow: bool
    mem_access: MemAccess
    flags: FlagEffects
    operands: tuple = ()
    # reg-var pairs that must use different registers (CONSTRAINED UNPREDICTABLE if they alias),
    # e.g. ("t", "t2") for LDP's two destinations. Operands are matched by their asl_index.
    constraints: tuple = ()
