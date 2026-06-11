from __future__ import annotations
import re
import xml.etree.ElementTree as ET
from .models import Operand, OperandKind, ExtractionError

_MULTIPLE = re.compile(r"multiple of (\d+)")
_RANGE = re.compile(r"([+-]?\d+) to ([+-]?\d+)")
_CONST_DEF = re.compile(r"constant\s+(\w+)\s*:\s*[\w{}]+\s*=\s*(\d+)\s*;")


def architectural_constants(shared_pseudocode_path) -> dict:
    """ASL integer constants from shared_pseudocode.xml, e.g. {'LOG2_TAG_GRANULE': 4}."""
    txt = "\n".join("".join(p.itertext()) for p in ET.parse(shared_pseudocode_path).iter("pstext"))
    return {m.group(1): int(m.group(2)) for m in _CONST_DEF.finditer(txt)}


def _ror(x: int, r: int, bits: int) -> int:
    r %= bits
    return ((x >> r) | (x << (bits - r))) & ((1 << bits) - 1)


def generate_logical_immediates(bits: int) -> tuple:
    """All valid ARM bitmask-immediate values for a *bits*-wide register (the DecodeBitMasks set)."""
    sizes = [2, 4, 8, 16, 32] if bits == 32 else [2, 4, 8, 16, 32, 64]
    out: set[int] = set()
    for size in sizes:
        for ones in range(1, size):
            pattern = (1 << ones) - 1
            for immr in range(size):
                rotated = _ror(pattern, immr, size)
                full = 0
                for _ in range(bits // size):
                    full = (full << size) | rotated
                out.add(full)
    return tuple(str(v) for v in sorted(out))


def _box_bits(box: ET.Element) -> list:
    """Bit values (MSB-first) of a regdiagram <box>: '0'/'1' fixed, else None (variable)."""
    w = box.get("width")
    width = int(w) if w is not None else 1            # ARM DTD: a box with no width is a single bit
    bits: list = []
    for c in box.findall("c"):
        v = c.text if c.text in ("0", "1") else None
        cs = c.get("colspan")
        bits += [v] * (int(cs) if cs is not None else 1)  # DTD: a <c> with no colspan covers 1 bit
    bits += [None] * (width - len(bits))
    return bits[:width]


def constant_fields(iclass_regdiagram: ET.Element, encoding: ET.Element) -> dict:
    """Per-encoding fully-constant field values: merge iclass bits with the encoding's pinned bits."""
    base = {b.get("name"): _box_bits(b) for b in iclass_regdiagram.findall("box") if b.get("name")}
    over = {b.get("name"): _box_bits(b) for b in encoding.findall("box") if b.get("name")}
    out = {}
    for name in set(base) | set(over):
        b = base[name] if name in base else []
        o = over[name] if name in over else []
        width = max(len(b), len(o))
        merged = [(o[i] if i < len(o) and o[i] is not None else (b[i] if i < len(b) else None))
                  for i in range(width)]
        if merged and all(x in ("0", "1") for x in merged):
            out[name] = int("".join(merged), 2)
    return out


def _statement(decode: str, pos: int) -> str:
    return decode[decode.rfind(";", 0, pos) + 1: pos]


def _resolve_scale(token: str, ctx, enc: str) -> int:
    token = token.strip()
    if re.fullmatch(r"\d+", token):
        return int(token)
    if token in ctx.arch_constants:                      # named ASL constant, e.g. LOG2_TAG_GRANULE
        return ctx.arch_constants[token]
    m = re.fullmatch(r"UInt\(\s*(\w+)\s*\)", token)
    if m is None:                                        # maybe a let-bound var: `let scale = UInt(size)`
        b = re.search(r"\b%s\b\s*(?::[^=]*)?=\s*UInt\(\s*(\w+)\s*\)" % re.escape(token), ctx.decode)
        if b is not None:
            m = b
    if m is not None and m.group(1) in ctx.const_fields:
        return ctx.const_fields[m.group(1)]
    raise ExtractionError(f"{enc}: cannot resolve immediate scale token {token!r}")


def _scale(field: str, ctx, enc: str) -> int:
    """log2(stride) from the Decode transform on <field>: LSL/<</* or unconditional concat-zeros."""
    f, decode = re.escape(field), ctx.decode
    m = (re.search(r"LSL\(\s*[^,]*\b%s\b[^,]*,\s*([^)]+)\)" % f, decode)
         or re.search(r"\b%s\b\s*\)?\s*<<\s*([^);\n]+)" % f, decode))
    if m is not None:
        return _resolve_scale(m.group(1), ctx, enc)
    mul = re.search(r"\b%s\b\s*\*\s*(\d+)" % f, decode) or re.search(r"(\d+)\s*\*\s*\b%s\b" % f, decode)
    if mul is not None:
        c = int(mul.group(1))
        if c & (c - 1):
            raise ExtractionError(f"{enc}: immediate {field!r} scaled by non-power-of-2 {c}")
        return c.bit_length() - 1
    cc = re.search(r"\b%s\b\s*::\s*(?:'(0+)'|Zeros\{(\d+)\})" % f, decode)
    if cc is not None and "if " not in _statement(decode, cc.start()):  # conditional concat => separate shift operand
        return len(cc.group(1)) if cc.group(1) else int(cc.group(2))
    return 0


def _signed(field: str, decode: str) -> bool:
    return re.search(r"(?:SignExtend|SInt)\w*\{?\d*\}?\(\s*[^)]*\b%s\b" % re.escape(field), decode) is not None


def _offset(field: str, decode: str) -> int:
    m = re.search(r"UInt\(\s*%s\s*\)\s*\+\s*(\d+)\b" % re.escape(field), decode)
    return int(m.group(1)) if m is not None else 0


_UNDEF_GUARD = re.compile(
    r"if\s+(.+?)\s+then\s+(?:EndOfDecode\(\s*Decode_UNDEF\s*\)|UNDEFINED|ReservedValue\(\s*\))")


def _effective_width(field: str, ctx, enc: str) -> int:
    """Box width of <field> minus the high bits a Decode UNDEF guard forces to 0, e.g.
    `if sf == '0' && imm6[5] == '1' then UNDEF`: with sf pinned to 0 in this encoding the guard
    forces imm6's MSB to 0, so a 32-bit shift amount is [0,31] (5 bits), not the box's [0,63]."""
    width = ctx.boxes[field]
    fbit = re.compile(r"%s\s*[\[<]\s*(\d+)\s*[\]>]\s*==\s*'([01])'" % re.escape(field))
    cfield = re.compile(r"(\w+)\s*==\s*'([01]+)'")
    forced: dict[int, int] = {}
    for g in _UNDEF_GUARD.finditer(ctx.decode):
        target = None
        rest_const_true = True
        for cond in (c.strip() for c in g.group(1).split("&&")):
            mf = fbit.fullmatch(cond)
            if mf is not None:
                target = (int(mf.group(1)), int(mf.group(2)))
                continue
            mc = cfield.fullmatch(cond)
            if (mc is not None and mc.group(1) in ctx.const_fields
                    and ctx.const_fields[mc.group(1)] == int(mc.group(2), 2)):
                continue            # conjunct is constant-true for this encoding
            rest_const_true = False  # a runtime/unknown conjunct => guard not unconditional here
        if target is not None and rest_const_true:
            bit_index, undef_value = target
            forced[bit_index] = 1 - undef_value  # valid encodings need the complement of the UNDEF bit
    if not forced:
        return width
    top = set(range(width - len(forced), width))
    if set(forced) == top and all(v == 0 for v in forced.values()):
        return width - len(forced)   # contiguous high zeros => simply a narrower field
    raise ExtractionError(f"{enc}: field {field!r} UNDEF bit-constraint {forced} is not a high-zero pattern")


def _is_label(field: str, ctx) -> bool:
    """PC-relative target (branch offset / ADR / ADRP): the field flows into a value that is
    added to / subtracted from PC64() (or a PC64()-derived base). Distinguishes a label from a
    base-register memory offset (added via AddressAdd(address, ...))."""
    text = ctx.decode + "\n" + ctx.execute
    subs = re.split(r"\s*::\s*", field.strip("()"))
    assign = re.compile(r"(?:let|var)\s+(\w+)\b[^=;]*=\s*([^;]*)")
    holders = {m.group(1) for m in assign.finditer(text)
               if any(re.search(r"\b%s\b" % re.escape(s), m.group(2)) for s in subs)}
    pc_bases = [r"PC64\(\)"] + [r"\b%s\b" % re.escape(m.group(1))
                                for m in assign.finditer(text) if "PC64()" in m.group(2)]
    toks = [re.escape(t) for t in set(subs) | holders]
    for base in pc_bases:                       # field/holder on one side of `base +/- ...`
        for tok in toks:
            if re.search(base + r"\s*[-+][^;,)]*\b" + tok + r"\b", text) or \
               re.search(r"\b" + tok + r"\b[^;,(]*[-+]\s*" + base, text):
                return True
    return False


def _field_names(field: str) -> list:
    """The encoding box(es) backing an immediate: one box (`imm12`) or a concat (`(immhi::immlo)`)."""
    return [s for s in re.split(r"\s*::\s*", field.strip("()")) if s]


def _field_width(field: str, ctx, enc: str) -> int:
    names = _field_names(field)
    for n in names:
        if n not in ctx.boxes:
            raise ExtractionError(f"{enc}: immediate field {field!r} part {n!r} is not an encoding box")
    if len(names) == 1:
        return _effective_width(names[0], ctx, enc)   # single field: honor Decode UNDEF bit-constraints
    return sum(ctx.boxes[n] for n in names)           # concatenated fields: total bit width


def _prose_range(prose: str):
    """The assembly value set as the operand's prose states it: (lo, hi, stride), or None if absent.
    Prose is authoritative for assembly values because the Decode ASL transform can diverge from what
    the assembler accepts (EXTQ computes imm<<3 internally yet the syntax takes the raw imm; LD1D scales
    by 8 in the addressing, invisible in Decode). The bit width then cross-checks the value count."""
    r = _RANGE.search(prose)
    if r is None:
        return None
    m = _MULTIPLE.search(prose)
    return (int(r.group(1)), int(r.group(2)), int(m.group(1)) if m is not None else 1)


# SVE tsz-encoded shifts: prose "N to number of bits per element [minus 1]"; the upper bound is the
# element size, selected at run time by the tsz bits of the same field (esize = 8 << HighestSetBitNZ).
_BPE = re.compile(r"(\d+)\s+to\s+(?:the\s+)?number of bits per element(\s+minus\s+1)?", re.I)
_TSIZE_BITS = re.compile(r"\btsize\s*:\s*bits\((\d+)\)")


def _bits_per_element_ranges(prose, field, width, ctx, enc) -> tuple:
    """(esize, lo, hi, stride) per element size for a tsz-encoded shift amount."""
    bpe = _BPE.search(prose)
    lo = int(bpe.group(1))
    minus_one = bpe.group(2) is not None
    tw = _TSIZE_BITS.search(ctx.decode)
    if tw is None:
        raise ExtractionError(f"{enc}: immediate {field!r} is 'bits per element' but Decode has no tsize width")
    tsize_bits = int(tw.group(1))
    esizes = [8 << k for k in range(tsize_bits)]          # esize = 8 << HighestSetBitNZ(tsize), HSB in [0,tsize_bits)
    ranges = tuple((e, lo, (e - 1) if minus_one else e, 1) for e in esizes)
    total = sum(hi - lo + 1 for _, _, hi, _ in ranges)
    valid = (1 << width) - (1 << (width - tsize_bits))   # tsize == 0 is UNDEF
    if total != valid:
        raise ExtractionError(
            f"{enc}: immediate {field!r} bits-per-element gives {total} values != {valid} valid encodings")
    return ranges


def immediate_operand(name, field, prose, ctx, datasize, enc) -> Operand:
    if _is_label(field, ctx):
        return Operand(name=name, kind=OperandKind.LABEL, read=True, write=False, width=0, signed=True)
    if "DecodeBitMasks" in ctx.decode:
        if not datasize:
            raise ExtractionError(f"{enc}: bitmask immediate {name!r} but no datasize")
        return Operand(name=name, kind=OperandKind.IMM, read=True, write=False,
                       width=datasize, signed=False, values=generate_logical_immediates(datasize))
    width = _field_width(field, ctx, enc)
    encodings = 1 << width                               # number of distinct valid field bit-patterns
    if _BPE.search(prose) is not None:                   # value set coupled to element size (SVE tsz shift)
        ranges = _bits_per_element_ranges(prose, field, width, ctx, enc)
        return Operand(name=name, kind=OperandKind.IMM, read=True, write=False,
                       width=width, signed=False, imm_ranges=ranges)
    prose_rng = _prose_range(prose)
    if prose_rng is not None:
        lo, hi, stride = prose_rng
        if (hi - lo) % stride != 0:
            raise ExtractionError(f"{enc}: immediate {name!r} prose range {prose_rng} is not a whole number of strides")
        values = (hi - lo) // stride + 1
        if values > encodings:                           # the value set must fit the bit budget (fewer => high encodings reserved)
            raise ExtractionError(
                f"{enc}: immediate {name!r} prose {prose_rng} = {values} values exceeds 2^{width} = {encodings} encodings")
        return Operand(name=name, kind=OperandKind.IMM, read=True, write=False,
                       width=width, signed=lo < 0, imm_ranges=((0, lo, hi, stride),))
    if len(_field_names(field)) > 1:
        raise ExtractionError(f"{enc}: multi-field immediate {name!r} has no prose range to define its values")
    signed = _signed(field, ctx.decode)                  # no prose range: deterministic from encoding+ASL
    stride = 1 << _scale(field, ctx, enc)
    mult = _MULTIPLE.search(prose)
    if mult is not None and int(mult.group(1)) != stride:  # prose multiple without a range still cross-checks stride
        raise ExtractionError(f"{enc}: immediate {name!r} encoding stride {stride} != prose 'multiple of {mult.group(1)}'")
    base = _offset(field, ctx.decode)
    lo = (-(1 << (width - 1)) if signed else 0) * stride + base
    hi = ((1 << (width - 1)) - 1 if signed else (1 << width) - 1) * stride + base
    return Operand(name=name, kind=OperandKind.IMM, read=True, write=False,
                   width=width, signed=signed, imm_ranges=((0, lo, hi, stride),))
