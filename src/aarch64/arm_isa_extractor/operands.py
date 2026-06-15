from __future__ import annotations
import re
from dataclasses import replace
import xml.etree.ElementTree as ET
from .models import Operand, OperandKind, MemAccess, MemRole, RegFile, ExtractionError
from .asl import AslSemantics
from .immediate import immediate_operand

# asm-token prefix -> (register file, width). Width 0 = variable (V vector / SVE scalable).
_REG_PREFIX = {
    "W": (RegFile.GP, 32), "X": (RegFile.GP, 64),
    "B": (RegFile.SIMD, 8), "H": (RegFile.SIMD, 16), "S": (RegFile.SIMD, 32),
    "D": (RegFile.SIMD, 64), "Q": (RegFile.SIMD, 128), "V": (RegFile.SIMD, 0),
    "Z": (RegFile.SVE_Z, 0), "P": (RegFile.SVE_P, 0),
}
_COND_CODES = frozenset("eq ne cs cc mi pl vs vc hi ls ge lt gt le al nv".split())
_EXTEND_OPS = frozenset("lsl lsr asr ror uxtb uxth uxtw uxtx sxtb sxth sxtw sxtx msl".split())
_PLACEHOLDER = frozenset({"reserved", "unallocated"})   # ARM marks unallocated table rows, not real values


def _tokens(asm: str):
    """Ordered (symbol, bracket) from an asmtemplate. `bracket` is the index (1, 2, ...) of the memory
    `[...]` the token sits in, so two regions (`[<Xd>], [<Xs>]`) are told apart, else 0.

    Only a STANDALONE `[...]` is a memory address: it is separated (a space or comma) from what
    precedes it — a comma for a later operand, the mnemonic gap for a first operand (MOPS `SET [<Xd>]!`).
    A `[...]` ATTACHED to the register it indexes (no separator) is a lane/element/tile index, not an
    address (e.g. `<Vt>.D[<index>]`, `ZA.D[<Wv>, <offs>]`); its tokens get bracket 0, like any operand."""
    out, depth, bracket, current, i = [], 0, 0, 0, 0
    while i < len(asm):
        c = asm[i]
        if c == "[":
            if depth == 0:
                if i == 0 or asm[i - 1] in " \t,":  # separated -> a memory address bracket
                    bracket += 1
                    current = bracket
                else:                               # abuts a register -> element/lane index
                    current = 0
            depth += 1; i += 1
        elif c == "]":
            depth -= 1; i += 1
        elif c == "<":
            j = asm.index(">", i)
            out.append((asm[i:j + 1], current if depth > 0 else 0)); i = j + 1
        else:
            i += 1
    return out


def _explanation_for(symbol: str, encoding_name: str, explanations: list[ET.Element]):
    for ex in explanations:
        sym = ex.find("symbol")
        if sym is None or "".join(sym.itertext()) != symbol:
            continue
        enclist = ex.get("enclist")
        if enclist is None:
            continue
        if encoding_name in [e.strip() for e in enclist.split(",")]:
            return ex
    return None


def _regvar(field: str) -> str | None:
    """Encoding field -> ASL reg-number variable: strip the file letters, e.g. Rd->d, Zda->da, Pg->g."""
    m = re.fullmatch(r"[A-Z]+([a-z][a-z0-9]*)", field)
    return m.group(1) if m else None


def _regvar_from_decode(field: str, decode: str) -> str | None:
    """Reg-number variable for a multi-field register: the Decode `let v = UInt(<fields>)` binding,
    e.g. `(T::Zt)` -> `let t : integer = UInt(T::'00'::Zt)` -> 't'."""
    subs = [s for s in re.split(r"\s*::\s*", field.strip("()")) if s]
    for m in re.finditer(r"let\s+(\w+)\s*(?::[^=]*)?=\s*UInt\(([^)]*)\)", decode):
        if all(re.search(r"\b%s\b" % re.escape(s), m.group(2)) for s in subs):
            return m.group(1)
    return None


def _reg_spec(symbol: str):
    s = symbol.strip("<>").split("|")[0]
    return _REG_PREFIX.get(s[:1]) if s else None


# accessor File letter (in the ASL) -> register file; the width of a `<R><t>`-style composite is
# variable (the `<R>` selector picks W/X), so it is 0.
_FILE_OF_ACCESSOR = {"X": RegFile.GP, "W": RegFile.GP, "V": RegFile.SIMD, "Q": RegFile.SIMD,
                     "D": RegFile.SIMD, "S": RegFile.SIMD, "H": RegFile.SIMD, "B": RegFile.SIMD,
                     "Z": RegFile.SVE_Z, "P": RegFile.SVE_P}


def _composite_reg_spec(var: str, asl: str):
    """(file, width) for a register written as two adjacent tokens `<R><t>`, where the asm gives no
    file letter (the width is in `<R>`). The file comes from the ASL accessor of the var, e.g.
    `X{datasize}(t)` -> GP, or `ExtendReg{}(m,...)`/`ShiftReg{}(m,...)` (always GP). Width is 0 (variable)."""
    m = re.search(r"\b([XWVQDSHBZP])(?:\{[^}]*\})?\(\s*\(?\s*%s\b" % re.escape(var), asl)
    if m is not None:
        return (_FILE_OF_ACCESSOR[m.group(1)], 0)
    if re.search(r"(?:ExtendReg|ShiftReg)(?:\{[^}]*\})?\(\s*%s\b" % re.escape(var), asl):
        return (RegFile.GP, 0)
    return None


_BIT_LITERAL = re.compile(r"'([01]+)'")


def _reserves_all_ones(fields, decode) -> bool:
    """True if the Decode UNDEFs the all-ones value of one of *fields* (e.g. `if Rm == '11111' then
    UNDEF`): then that field cannot be its top value, so the register's highest number is reserved."""
    for f in fields:
        if re.search(r"\bif\s+%s\s*==\s*'1+'\s*then[^;]*(?:Decode_UNDEF|UNDEFINED|ReservedValue)"
                     % re.escape(f), decode):
            return True
    return False


def _register_range(var, field, ctx, enc) -> tuple:
    """Encodable register numbers (lo, hi, stride) for reg-number var <var>. The field width gives
    the full range; fixed bits in the Decode `let var = UInt(<concat>)` binding restrict it
    (e.g. `Rm::'0'` => even regs, `'011'::Rm` => a fixed high block), and a reserved all-ones field
    drops the top number. Loud-fails on bindings that are not a clean prefix/field/suffix concat."""
    m = re.search(r"\blet\s+%s\b\s*:\s*integer\b[^=]*=\s*UInt\(\s*([^)]*)\)" % re.escape(var), ctx.decode)
    if m is None:                                # no transform binding: full width range
        names = [f for f in re.split(r"\s*::\s*", field.strip("()")) if f]
        hi = (1 << sum(ctx.boxes[f] for f in names)) - 1
        return (0, hi - (len(names) == 1 and _reserves_all_ones(names, ctx.decode)), 1)
    parts = []                                   # (bits, value-or-None, name) MSB->LSB; None = varying field
    for tok in (t.strip() for t in re.split(r"\s*::\s*", m.group(1)) if t.strip()):
        lit = _BIT_LITERAL.fullmatch(tok)
        if lit is not None:
            parts.append((len(lit.group(1)), int(lit.group(1), 2), None))
        elif tok in ctx.boxes:
            parts.append((ctx.boxes[tok], None, tok))
        else:
            raise ExtractionError(f"{enc}: register {var!r} binding token {tok!r} not a bit-literal or field")
    varying = [i for i, p in enumerate(parts) if p[1] is None]
    if not varying:
        raise ExtractionError(f"{enc}: register {var!r} binding has no varying field")
    first, last = varying[0], varying[-1]
    if any(parts[i][1] is not None for i in range(first, last + 1)):
        raise ExtractionError(f"{enc}: register {var!r} has fixed bits between fields (not a simple range)")
    var_bits = sum(parts[i][0] for i in range(first, last + 1))
    suffix_bits = sum(b for b, _, _ in parts[last + 1:])
    base = 0                                     # fixed prefix and suffix bits frame the varying block
    for b, v, _ in parts[:first]:                # prefix fixed bits (high)
        base = (base << b) | v
    base <<= var_bits                            # leave the varying block at 0 (the low end of the range)
    for b, v, _ in parts[last + 1:]:             # suffix fixed bits (low)
        base = (base << b) | v
    stride = 1 << suffix_bits
    hi = base + stride * ((1 << var_bits) - 1)
    if len(varying) == 1 and _reserves_all_ones([parts[first][2]], ctx.decode):
        hi -= stride                             # the single varying field's all-ones (the top reg) is reserved
    return (base, hi, stride)


def _reg_operand(symbol, var, field, ctx, file, width, sem, encoding_name) -> Operand:
    # A register's role comes from the ASL whether or not it sits inside a `[...]`: a memory base is
    # read (and written under writeback), an index is read. The memory access itself is the
    # instruction's mem_access, not the register's read/write.
    read, write = var in sem.read_regvars, var in sem.written_regvars
    if not read and not write:   # a real register operand is always read and/or written;
        raise ExtractionError(    # neither means the ASL did not reveal its role (no silent-wrong)
            f"{encoding_name}: register {symbol!r} (var {var!r}) role undetermined from ASL")
    return Operand(name=symbol.strip("<>"), kind=OperandKind.REG, read=read, write=write, width=width,
                   signed=False, reg_file=file, asl_index=var, sp_capable=var in sem.sp_regvars,
                   reg_range=_register_range(var, field, ctx, encoding_name))


def _enum_value(sym: str):
    """A definition-table symbol -> its real value, or None to drop it. Drops the non-values: a
    sub-token (`<x>`), an unallocated-row marker (RESERVED/UNALLOCATED), and an immediate-syntax
    template (`#uimm5`). A `#`-number is a real enumerated immediate -> its number (`#0.5`->"0.5",
    `#90`->"90"): a value starts with a digit/sign/dot, a template with a letter."""
    v = sym.lower()
    if v.startswith("<") or v in _PLACEHOLDER:
        return None
    if v.startswith("#"):
        return v[1:] if v[1:2] in "0123456789+-." else None
    return v


def _enumerated(symbol, definition) -> Operand:
    tbl = definition.find(".//table")
    syms = [e.text for e in tbl.iter("entry")
            if e.get("class") == "symbol" and e.text] if tbl is not None else []
    # a single table cell may list display aliases of one encoding joined by '|' (e.g. an extend
    # `LSL|UXTW` for the same option value); each alias is an independent valid mnemonic, so split them.
    vals = tuple(part for v in (_enum_value(s) for s in syms) if v is not None for part in v.split("|"))
    if vals and set(vals) <= _COND_CODES:
        kind = OperandKind.COND
    elif vals and set(vals) <= _EXTEND_OPS:
        kind = OperandKind.EXTEND
    else:
        kind = OperandKind.IMM
    return Operand(name=symbol.strip("<>"), kind=kind, read=True, write=False, width=0, signed=False, values=vals)


def _composite_pair(asm: str):
    """Two adjacent `<...><...>` tokens (no separator) are one register, e.g. TBZ's `<R><t>`: the
    first is the width selector, the second the register number. Returns (selectors, registers)."""
    pairs = re.findall(r"(<[^>]+>)(<[^>]+>)", asm)
    return {p[0] for p in pairs}, {p[1] for p in pairs}


def _writeback_targets(asm: str) -> set:
    """Operand tokens whose register the asm updates (writeback). The base register is written by
    pre-index `[<Xn>, #<imm>]!`, post-index `[<Xn>], #<imm>`, or `[<Xd>]!`; a standalone `<Xn>!` is
    too. Other addressing forms (offset/unsigned `[<Xn>{, #<imm>}]`) do not write the base."""
    standalone = {m.group(1) for m in re.finditer(r"<([^>]+)>!", asm)}
    pre_index = {m.group(1) for m in re.finditer(r"\[\s*<([^>]+)>[^\]]*\]!", asm)}
    post_index = {m.group(1) for m in re.finditer(r"\[\s*<([^>]+)>[^\]]*\]\s*,", asm)}
    return standalone | pre_index | post_index


def _reg_file_width(symbol, var, encodedin, is_composite, ctx, enc):
    """(file, width) if the token is a register, else None. The `<R><t>` second half is always a
    register (its file is read from the ASL); a plain token's file comes from its asm letter."""
    if is_composite:
        if var is None:
            raise ExtractionError(f"{enc}: composite register {symbol!r} (field {encodedin!r}) reg-var not resolvable")
        spec = _composite_reg_spec(var, ctx.execute)
        if spec is None:
            raise ExtractionError(f"{enc}: composite register {symbol!r} (var {var!r}) file not found in ASL")
        return spec
    return _reg_spec(symbol)


def _addressing_role(op, base_seen) -> MemRole:
    """The role of an operand sitting inside a memory `[...]`: first register is the base, later
    registers are the index, a shift/extend type is EXTEND, anything else is the offset."""
    if op.kind is OperandKind.REG:
        return MemRole.BASE if not base_seen else MemRole.INDEX
    if op.kind is OperandKind.EXTEND:
        return MemRole.EXTEND
    return MemRole.OFFSET


def build_operands(encoding_name: str, asm: str, explanations: list[ET.Element],
                   sem: AslSemantics, ctx) -> tuple:
    selectors, composite_regs = _composite_pair(asm)
    writeback = _writeback_targets(asm)
    ops, datasize, base_bracket = [], None, 0   # base_bracket = the bracket whose base reg we've seen
    for symbol, bracket in _tokens(asm):
        if symbol in selectors:
            continue                            # the width selector is folded into its register
        # a token addresses memory iff it sits in a memory `[...]` (bracket > 0; _tokens excludes
        # register element/lane indices) of a memory-accessing instruction.
        is_addr = sem.mem_access is not MemAccess.NONE and bracket > 0
        ex = _explanation_for(symbol, encoding_name, explanations)
        if ex is None:
            raise ExtractionError(f"{encoding_name}: no explanation for operand {symbol!r}")
        account = ex.find("account")
        definition = ex.find("definition")
        if account is not None:
            encodedin = account.get("encodedin")
            if not encodedin:
                raise ExtractionError(f"{encoding_name}: operand {symbol!r} account has no encodedin")
            var = _regvar_from_decode(encodedin, ctx.decode) if "::" in encodedin else _regvar(encodedin)
            spec = _reg_file_width(symbol, var, encodedin, symbol in composite_regs, ctx, encoding_name)
            if spec is None:
                op = immediate_operand(symbol.strip("<>"), encodedin, "".join(account.itertext()),
                                       ctx, datasize, encoding_name)
            else:
                if var is None:
                    raise ExtractionError(
                        f"{encoding_name}: register {symbol!r} (field {encodedin!r}) reg-var not resolvable")
                file, width = spec
                op = _reg_operand(symbol, var, encodedin, ctx, file, width, sem, encoding_name)
                if not is_addr and file == RegFile.GP and width:
                    datasize = width            # the data register's width (e.g. bitmask datasize)
        elif definition is not None:
            op = _enumerated(symbol, definition)
        else:
            raise ExtractionError(f"{encoding_name}: operand {symbol!r} has neither account nor definition")
        if is_addr:
            # first register of each bracket is its base; a later register in the SAME bracket is an
            # index (so two regions `[<Xd>], [<Xs>]` give two bases, not base+index).
            op = replace(op, mem_role=_addressing_role(op, bracket == base_bracket))
            if op.kind is OperandKind.REG:
                base_bracket = bracket
        if op.kind is OperandKind.REG:
            wb = op.name in writeback
            if op.mem_role in (MemRole.BASE, MemRole.INDEX):
                # an address register is written iff the form writes back (asm-precise); this also
                # overrides the shared Execute's `if wback` write, which otherwise leaks to the
                # non-writeback (offset/unsigned) encodings.
                op = replace(op, write=wb)
            elif wb and not op.write:
                op = replace(op, write=True)
        ops.append(op)
    return tuple(ops)
