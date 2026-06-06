"""
ARM ISA XML parser.

Parses one <instructionsection type="instruction"> at a time from a single
ARM ISA XML file and returns a list of InstructionSpec objects.

Design principles
-----------------
- Pure stdlib (no numpy, no subprocess).
- Every error is caught per-instruction; one bad entry never aborts a file.
- All static data lives in registers.py; the parser contains only logic.
- Merging of duplicate encodings (same instruction, different XML symbols for
  the same register class) is handled by the pipeline layer in pipeline.py,
  not here.
"""
from __future__ import annotations

import copy
import logging
import re
from math import log2
from pathlib import Path
from typing import Optional
from xml.etree import ElementTree as ET

from .models import InstructionSpec, OperandSpec
from . import registers as regs

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sentinel exception — expected failures that should be silently skipped
# ---------------------------------------------------------------------------

class ParseFailed(Exception):
    """Raised when a single encoding cannot be parsed and should be skipped."""


# ---------------------------------------------------------------------------
# ARM bitmask-immediate generator
# ---------------------------------------------------------------------------

def _ror(x: int, r: int, bits: int) -> int:
    r %= bits
    return ((x >> r) | (x << (bits - r))) & ((1 << bits) - 1)


def generate_logical_immediates(bits: int = 64) -> list[str]:
    """Return all valid ARM bitmask-immediate values for *bits*-wide registers."""
    sizes = [2, 4, 8, 16, 32] if bits == 32 else [2, 4, 8, 16, 32, 64]
    immediates: set[int] = set()
    for S in sizes:
        for ones in range(1, S):
            pattern = (1 << ones) - 1
            for immr in range(S):
                rotated = _ror(pattern, immr, S)
                full = 0
                for _ in range(bits // S):
                    full = (full << S) | rotated
                immediates.add(full)
    return [str(v) for v in sorted(immediates)]


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class InstructionFileParser:
    """
    Parse all instructions from a single ARM ISA XML file.

    Parameters
    ----------
    path:
        Path to the XML file.
    allowed_categories:
        If provided, only instructions whose ``instr-class`` docvar is in this
        set are emitted.  Pass ``None`` (default) to accept all categories.
    allowed_address_forms:
        Set of ``address-form`` docvar values to accept.
    """

    _DEFAULT_ADDRESS_FORMS: frozenset[str] = frozenset(
        {"literal", "base-register", "post-indexed", "pre-indexed", "unsigned-scaled-offset", ""}
    )

    def __init__(
        self,
        path: Path,
        *,
        allowed_categories: Optional[set[str]] = None,
        allowed_address_forms: Optional[frozenset[str]] = None,
    ) -> None:
        self.path = path
        self.allowed_categories = allowed_categories
        self.allowed_address_forms = (
            allowed_address_forms
            if allowed_address_forms is not None
            else self._DEFAULT_ADDRESS_FORMS
        )

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def parse(self) -> list[InstructionSpec]:
        """Parse the file and return all valid InstructionSpec objects."""
        try:
            tree = ET.parse(self.path)
        except ET.ParseError as exc:
            logger.error("XML parse error in %s: %s", self.path.name, exc)
            return []

        results: list[InstructionSpec] = []
        for section in tree.iter("instructionsection"):
            if section.get("type") != "instruction":
                continue
            explanations = section.find(".//explanations")
            if explanations is None:
                continue
            try:
                results.extend(self._parse_section(section, explanations))
            except Exception as exc:
                logger.warning(
                    "Unhandled error in %s: %s", self.path.name, exc, exc_info=True
                )
        return results

    # ------------------------------------------------------------------
    # Section → list[InstructionSpec]
    # ------------------------------------------------------------------

    def _parse_section(
        self, section: ET.Element, explanations: ET.Element
    ) -> list[InstructionSpec]:
        docvars = section.find("docvars")
        if docvars is None:
            return []

        category = _get_docvar(docvars, "instr-class")
        if self.allowed_categories and category not in self.allowed_categories:
            return []

        flags_op = _extract_flags_operand(section)
        results: list[InstructionSpec] = []

        for encoding in section.findall("classes/iclass/encoding"):
            try:
                results.extend(self._parse_encoding(encoding, explanations, flags_op))
            except ParseFailed as exc:
                logger.debug("Skipped encoding in %s: %s", self.path.name, exc)
            except Exception as exc:
                logger.warning(
                    "Unhandled error for encoding in %s: %s",
                    self.path.name, exc, exc_info=True,
                )
        return results

    # ------------------------------------------------------------------
    # Encoding element → list[InstructionSpec]
    # ------------------------------------------------------------------

    def _parse_encoding(
        self,
        encoding: ET.Element,
        explanations: ET.Element,
        flags_op: Optional[OperandSpec],
    ) -> list[InstructionSpec]:
        docvars = encoding.find("docvars")
        if docvars is None:
            raise ParseFailed("No <docvars> on encoding element")

        spec = InstructionSpec()
        address_form = ""

        for dv in docvars.iter("docvar"):
            key, val = dv.get("key", ""), dv.get("value", "")
            if key == "instr-class":
                spec.category = val
            elif key == "branch-offset":
                spec.control_flow = True
            elif key == "address-form":
                address_form = val
            elif key == "datatype":
                spec.datatype = val

        if address_form not in self.allowed_address_forms:
            raise ParseFailed(f"address-form={address_form!r} not allowed")

        mnemonic_node = encoding.find("asmtemplate/text")
        if mnemonic_node is None or not mnemonic_node.text:
            raise ParseFailed("Missing mnemonic <text> node")
        spec.name = mnemonic_node.text.split()[0]

        if spec.control_flow:
            spec.implicit_operands.append(
                OperandSpec("pc", "REG", ["PC"], src=True, dest=False, width=64, signed=False)
            )
        if flags_op:
            spec.implicit_operands.append(copy.deepcopy(flags_op))

        variant_name = encoding.get("name", "")
        template_text = _collect_asmtemplate_text(encoding)
        raw = self._expand_template(variant_name, template_text, spec, explanations)
        return self._split_mixed_width_specs(raw)

    # ------------------------------------------------------------------
    # Width-split pass
    # ------------------------------------------------------------------

    def _split_mixed_width_specs(
        self, specs: list[InstructionSpec]
    ) -> list[InstructionSpec]:
        """
        For instructions where the register width is chosen by the programmer
        at assembly time (TBZ/TBNZ being the canonical example), the XML may
        produce a single InstructionSpec whose register operand contains *both*
        32-bit and 64-bit registers, with a correspondingly wide immediate range
        that is only valid for one of the two widths.

        This pass detects such mixed-width specs for known instructions and
        splits each one into a pair of specs — one pure-32-bit and one
        pure-64-bit — applying the correct immediate constraints to each half.

        Design note: the split is done here, after template expansion, because
        at this point each spec is fully built (all operands resolved) and we
        can reason about the complete operand set.  Doing it inside the
        expansion loop would require threading width-awareness through the
        entire recursive descent.
        """
        result: list[InstructionSpec] = []
        for spec in specs:
            if spec.name.lower() in _WIDTH_SPLIT_MNEMONICS:
                result.extend(_split_by_register_width(spec))
            else:
                result.append(spec)
        return result

    # ------------------------------------------------------------------
    # Template expansion  (handles {optional}, (a|b), <operand>, [mem], #imm)
    # ------------------------------------------------------------------

    def _expand_template(
        self,
        variant_name: str,
        template: str,
        spec: InstructionSpec,
        explanations: ET.Element,
        *,
        mem_depth: int = 0,
        offset: int = 0,
        width_prefix: Optional[str] = None,
    ) -> list[InstructionSpec]:
        """
        Walk *template* from *offset*, building up *spec* in place.
        When optional groups ``{}`` or alternatives ``(a|b)`` are encountered,
        the spec is deep-copied and two (or more) recursive calls are made,
        returning the union of all valid encodings.

        width_prefix:
            When set to ``'W'`` or ``'X'``, the next ``<t>``-style register-
            number variable encountered is synthesised directly as a GP register
            operand of that width (w0–w30 or x0–x30) rather than going through
            the XML explanation lookup.  This is needed for instructions like
            TBZ/TBNZ whose assembly syntax is ``{R}<t>`` — where ``{R}`` is an
            ARM width-selector token, not a literal optional group — and whose
            XML explanation describes ``<t>`` as a bare integer 0–30 without
            specifying the register file.
        """
        SPECIAL = set("<>!|(){}#[]")
        hint = "REG"
        data_size_hint: Optional[int] = None
        i = offset

        while i < len(template):
            ch = template[i]

            # --- plain character ---
            if ch not in SPECIAL:
                spec.template += ch
                i += 1
                continue

            # --- operand: <VarName> ---
            if ch == "<":
                close = _find_close(template, i)
                variable = template[i + 1: close]

                # Width-prefix mode: the immediately following register-number
                # variable is synthesised from the prefix rather than looked up
                # in the XML explanations.  The compound name written into the
                # template is e.g. "Rt" (prefix + variable), matching standard
                # AArch64 assembly notation.
                if width_prefix is not None and _is_register_index_variable(variable):
                    compound_name = width_prefix + variable   # e.g. "Wt" or "Xt"
                    spec.template += f"{{{compound_name}}}"
                    reg_width = 32 if width_prefix == "W" else 64
                    reg_values = list(regs.GP_REGISTERS[reg_width])
                    operand = OperandSpec(
                        compound_name, "REG", reg_values,
                        src=True, dest=False, width=reg_width, signed=False,
                    )
                    if mem_depth > 0:
                        operand.type_ = "MEM"
                    else:
                        data_size_hint = reg_width
                    spec.operands.append(operand)
                    width_prefix = None   # consumed; subsequent variables resolve normally
                    i = close + 1
                    if mem_depth == 0:
                        hint = "REG"
                    continue

                spec.template += f"{{{variable}}}"

                # Before the normal explanation lookup, check whether this
                # variable is a W/X width-selector table (e.g. <R> in TBZ).
                # If it is, we do NOT emit an operand for it; instead we split
                # into two branches — one per prefix — and carry width_prefix
                # into the continuation so the immediately following <t>-style
                # variable is synthesised as a concrete GP register operand.
                # We also strip the spurious "{R}" we just appended to the
                # template, because the compound name (e.g. "Wt") written by
                # the width_prefix path will replace it.
                width_sel = _is_width_selector_for_variable(
                    spec.name, explanations, variant_name, variable
                )
                if width_sel is not None:
                    # Undo the "{R}" we already wrote into spec.template.
                    spec.template = spec.template[: -(len(variable) + 2)]
                    prefixes = ["W", "X"] if width_sel == "R" else [width_sel]
                    rest = template[close + 1:]
                    results: list[InstructionSpec] = []
                    for pfx in prefixes:
                        results += self._expand_template(
                            variant_name, rest, copy.deepcopy(spec), explanations,
                            mem_depth=mem_depth, offset=0, width_prefix=pfx,
                        )
                    return results

                operand = self._resolve_operand(
                    spec.name, explanations, variant_name, variable, hint, data_size_hint
                )
                if mem_depth > 0:
                    operand.type_ = "MEM"
                else:
                    # first non-memory operand tells us the data width for
                    # subsequent bitmask-immediate lookups
                    if data_size_hint is None:
                        data_size_hint = operand.width or data_size_hint
                spec.operands.append(operand)
                i = close + 1
                # reset hint after leaving a memory context
                if mem_depth == 0:
                    hint = "REG"

            # --- optional group, literal register list, or width-prefix token ---
            elif ch == "{":
                close = _find_close(template, i)
                inner = template[i + 1: close].strip()

                # { <Zt1>, <Zt2> } — literal braces in the assembly syntax,
                # not an optional group. Emit the opening brace, then parse
                # each <variable> inside normally so operands are resolved,
                # then emit the closing brace.
                if inner.startswith("<"):
                    spec.template += "{"
                    # Parse the interior as a sub-template (same mem_depth/hint)
                    # We do this by recursing on just the inner slice, then
                    # appending whatever template text was built, merging operands.
                    inner_spec = copy.deepcopy(spec)
                    inner_spec.template = ""
                    inner_results = self._expand_template(
                        variant_name, template[i + 1: close], inner_spec,
                        explanations, mem_depth=mem_depth, offset=0,
                        width_prefix=width_prefix,
                    )
                    if inner_results:
                        # Take the first (and normally only) result — register
                        # lists don't contain optional groups
                        resolved = inner_results[0]
                        spec.template += resolved.template + "}"
                        # Merge any newly resolved operands into spec
                        existing_names = {op.name for op in spec.operands}
                        for op in resolved.operands:
                            if op.name not in existing_names:
                                spec.operands.append(op)
                    else:
                        spec.template += template[i + 1: close] + "}"
                    i = close + 1

                # {R}, {W}, or {X} — ARM register-width selector token.
                # Not an optional group: it is mandatory and selects the GP
                # register file.  Split into two branches (W=32-bit, X=64-bit)
                # and carry width_prefix into the continuation so the next
                # <t>-style variable is synthesised correctly.
                elif inner.upper() in _WIDTH_PREFIX_TOKENS:
                    prefixes = (
                        ["W", "X"] if inner.upper() == "R" else [inner.upper()]
                    )
                    rest = template[close + 1:]
                    results: list[InstructionSpec] = []
                    for pfx in prefixes:
                        results += self._expand_template(
                            variant_name, rest, copy.deepcopy(spec), explanations,
                            mem_depth=mem_depth, offset=0, width_prefix=pfx,
                        )
                    return results

                # { optional content } — expand into two variants:
                # one with the content, one without.
                else:
                    t_with    = template[:i] + template[i + 1: close] + template[close + 1:]
                    t_without = template[:i] + template[close + 1:]
                    return (
                        self._expand_template(variant_name, t_with,    copy.deepcopy(spec), explanations, mem_depth=mem_depth, offset=i, width_prefix=width_prefix)
                        + self._expand_template(variant_name, t_without, copy.deepcopy(spec), explanations, mem_depth=mem_depth, offset=i, width_prefix=width_prefix)
                    )

            # --- alternative group: (a|b|c) → one branch per option ---
            elif ch == "(":
                close = _find_close(template, i)
                results: list[InstructionSpec] = []
                for option in template[i + 1: close].split("|"):
                    new_t = template[:i] + option.strip() + template[close + 1:]
                    results += self._expand_template(
                        variant_name, new_t, copy.deepcopy(spec), explanations,
                        mem_depth=mem_depth, offset=i,
                    )
                return results

            # --- stray closing chars (should never appear outside their pair) ---
            elif ch in "})|":
                raise ParseFailed(f"Unexpected '{ch}' at position {i} in: {template!r}")

            # --- memory addressing: [ opens, ] closes ---
            elif ch == "[":
                spec.template += "["
                mem_depth += 1
                hint = "MEM"
                i += 1

            elif ch == "]":
                spec.template += "]"
                mem_depth -= 1
                if mem_depth < 0:
                    raise ParseFailed(f"Unmatched ']' in: {template!r}")
                if mem_depth == 0:
                    hint = "REG"
                i += 1

            # --- immediate prefix ---
            elif ch == "#":
                spec.template += "#"
                hint = "IMM"
                i += 1

            # --- write-back marker ---
            elif ch == "!":
                spec.template += "!"
                i += 1

            else:
                raise ParseFailed(f"Unexpected character {ch!r} in: {template!r}")

        return [spec] if _postprocess(spec) else []

    # ------------------------------------------------------------------
    # Operand resolution — look up the matching <explanation> clause
    # ------------------------------------------------------------------

    def _resolve_operand(
        self,
        mnemonic: str,
        explanations: ET.Element,
        variant_name: str,
        variable: str,
        hint: str,
        data_size_hint: Optional[int],
    ) -> OperandSpec:
        src, dest = _memory_access_roles(mnemonic, hint)

        for explanation in explanations.findall(".//explanation"):
            if not _matches_enclist(explanation.get("enclist", ""), variant_name):
                continue
            if not _contains_variable(explanation, variable):
                continue

            accounts = explanation.findall(".//account")
            tables   = explanation.findall(".//table")

            n_accounts, n_tables = len(accounts), len(tables)
            if n_accounts + n_tables != 1:
                raise ParseFailed(
                    f"Expected exactly one <account> or <table> for "
                    f"{variant_name}:{variable}, found {n_accounts} accounts "
                    f"and {n_tables} tables"
                )

            if n_tables == 1:
                values, hint, src, dest = _handle_table(explanation, tables[0])
                return OperandSpec(variable, hint, values, src, dest, 0, False)

            # n_accounts == 1
            values, hint, src, dest, width, signed = _handle_account(
                accounts[0], hint, src, dest,
                mnemonic, variable, variant_name, data_size_hint,
            )
            return OperandSpec(variable, hint, values, src, dest, width, signed)

        # No explanation matched — return an empty operand rather than crashing.
        # This lets the caller decide whether to keep the encoding.
        logger.debug(
            "No explanation found for %s:%s (variant=%s)", mnemonic, variable, variant_name
        )
        return OperandSpec(variable, hint, [], src or False, dest or False, 0, False)


# ---------------------------------------------------------------------------
# Width-prefix / register-index helpers
# ---------------------------------------------------------------------------

# The set of single-character tokens (upper-case) that indicate a GP register
# width selector.  "R" means "W or X" (caller's choice); "W" and "X" are
# explicit.  These appear both as <R> in the XML asmtemplate and as {R} in
# some other ARM documents.
_WIDTH_PREFIX_TOKENS: frozenset[str] = frozenset({"R", "W", "X"})

# Variable names that represent a bare register index (0–30 / ZR).
# When width_prefix is set and we encounter one of these, we synthesise a
# proper REG operand instead of doing an XML explanation lookup.
_REGISTER_INDEX_VARIABLES: frozenset[str] = frozenset({
    "t", "t2", "n", "m", "d", "a", "s",
    "t3", "t4", "dn",
})


def _is_register_index_variable(variable: str) -> bool:
    """Return True if *variable* is a bare register-index placeholder."""
    return variable in _REGISTER_INDEX_VARIABLES


def _is_width_selector_for_variable(
    mnemonic: str,
    explanations: ET.Element,
    variant_name: str,
    variable: str,
) -> Optional[str]:
    """
    Look up *variable* in *explanations* and return ``"R"`` (or ``"W"``/
    ``"X"``) if its definition is a W/X width-selector table, or ``None``
    if it is anything else.

    This is a lightweight pre-check used by the template expander to detect
    the <R> pattern (as in TBZ/TBNZ) before committing to a full operand
    resolution.
    """
    for explanation in explanations.findall(".//explanation"):
        if not _matches_enclist(explanation.get("enclist", ""), variant_name):
            continue
        if not _contains_variable(explanation, variable):
            continue
        return _is_width_selector_table(explanation)
    return None



def _is_width_selector_table(explanation: ET.Element) -> Optional[str]:
    """
    If *explanation* describes a W/X width-selector table (like <R> in TBZ),
    return ``"R"``.  Otherwise return None.

    The qualifying conditions are strict to avoid false positives on
    instructions like SUBS extended-register, where the ``option`` field is
    3 bits wide and maps 8 rows to only {W, X} symbols:

      1. Exactly 2 non-RESERVED rows in the tbody.
      2. The two symbol values are exactly {"W", "X"} (one each).
      3. The bitfield column has exactly 2 distinct values (confirming 1-bit
         encoding — rules out multi-bit fields that happen to collapse to W/X).

    All three conditions must hold simultaneously.
    """
    table = explanation.find(".//table")
    if table is None:
        return None
    tbody = table.find(".//tbody")
    if tbody is None:
        return None

    rows = tbody.findall(".//row")
    non_reserved_rows = []
    for row in rows:
        entries = row.findall('.//entry[@class="symbol"]')
        if not entries:
            continue
        sym = (entries[0].text or "").strip().upper()
        if sym not in ("", "RESERVED"):
            non_reserved_rows.append(row)

    # Condition 1: exactly 2 rows
    if len(non_reserved_rows) != 2:
        return None

    symbols = set()
    bitfields = set()
    for row in non_reserved_rows:
        sym_entries = row.findall('.//entry[@class="symbol"]')
        bit_entries = row.findall('.//entry[@class="bitfield"]')
        if sym_entries:
            symbols.add((sym_entries[0].text or "").strip().upper())
        if bit_entries:
            bitfields.add((bit_entries[0].text or "").strip())

    # Condition 2: symbols are exactly {W, X}
    if symbols != {"W", "X"}:
        return None

    # Condition 3: bitfield column has exactly 2 distinct values (1-bit field)
    if len(bitfields) != 2:
        return None

    return "R"


# ---------------------------------------------------------------------------
# Module-level helpers (stateless, no self needed)
# ---------------------------------------------------------------------------

def _get_docvar(docvars: ET.Element, key: str) -> str:
    for dv in docvars.iter("docvar"):
        if dv.get("key") == key:
            return dv.get("value", "")
    return ""


def _collect_asmtemplate_text(encoding: ET.Element) -> str:
    return "".join(p.text or "" for p in encoding.findall(".//asmtemplate/*"))


def _find_close(s: str, open_idx: int) -> int:
    """Return the index of the matching closing bracket for the opener at *open_idx*."""
    PAIRS = {"<": ">", "{": "}", "(": ")", "[": "]"}
    opener = s[open_idx]
    closer = PAIRS.get(opener)
    if closer is None:
        raise ParseFailed(f"{s[open_idx]!r} is not an opener")
    depth = 0
    for i in range(open_idx, len(s)):
        if s[i] == opener:
            depth += 1
        elif s[i] == closer:
            depth -= 1
            if depth == 0:
                return i
    raise ParseFailed(f"No closing {closer!r} for opener at index {open_idx} in {s!r}")


def _matches_enclist(enclist: str, variant_name: str) -> bool:
    return any(variant_name.strip() == item.strip() for item in enclist.split(","))


def _contains_variable(explanation: ET.Element, variable: str) -> bool:
    return any(variable in (sym.text or "") for sym in explanation.findall(".//symbol"))


def _full_text(element: Optional[ET.Element]) -> str:
    """Concatenate all text content (including tails) within an element."""
    if element is None:
        return ""
    parts = [element.text or ""]
    for child in element:
        parts.append(child.text or "")
        parts.append(child.tail or "")
    return "".join(parts)


_STORE_PREFIXES: tuple[str, ...] = (
    "str", "stg", "stz", "st2", "stl", "stn", "stx", "stp", "stt", "stu",
)
_LOAD_PREFIXES: tuple[str, ...] = (
    "ldr", "ldg", "ldn", "ldp", "ldt", "ldu", "ldx", "lda",
)


def _memory_access_roles(
    mnemonic: str, hint: str
) -> tuple[Optional[bool], Optional[bool]]:
    """
    For load/store instructions, infer src/dest from the mnemonic and
    whether the current operand is inside a memory-address bracket.
    Returns (src, dest) or (None, None) when not a load/store.
    """
    m = mnemonic.lower()
    if any(m.startswith(p) for p in _STORE_PREFIXES):
        return hint != "MEM", hint == "MEM"
    if any(m.startswith(p) for p in _LOAD_PREFIXES):
        return hint == "MEM", hint != "MEM"
    return None, None


def _handle_table(
    explanation: ET.Element, table: ET.Element
) -> tuple[list[str], str, bool, bool]:
    """Parse a <table> element (condition codes or simple enum)."""
    intro = explanation.find(".//intro")
    hint = "IMM"
    src, dest = True, False

    if intro is not None and intro.text and "standard conditions" in intro.text:
        hint = "COND"
        src = dest = False

    values: list[str] = []
    tbody = table.find(".//tbody")
    if tbody is not None:
        for entry in tbody.findall('.//entry[@class="symbol"]'):
            if entry.text and entry.text != "RESERVED":
                values.append(entry.text)

    return values, hint, src, dest


def _handle_account(
    account: ET.Element,
    hint: str,
    src: Optional[bool],
    dest: Optional[bool],
    mnemonic: str,
    variable: str,
    variant_name: str,
    data_size_hint: Optional[int],
) -> tuple[list[str], str, bool, bool, int, bool]:
    """Parse an <account> element and return (values, hint, src, dest, width, signed)."""
    text = _full_text(account.find(".//para"))

    # Infer src/dest from prose when not already set by the mnemonic
    if src is None:
        src = any(w in text for w in ("loaded", "source", "tested"))
    if dest is None:
        dest = any(w in text for w in ("stored", "destination"))

    width = 0
    signed = False

    m_bits = re.search(r"(\d+)-bit", text)
    if m_bits:
        width = int(m_bits.group(1))

    # ---- label ----
    if "label" in text:
        return [], "LABEL", True, False, width, True

    # ---- general-purpose register ----
    if "general-purpose" in text and "register" in text and width in regs.GP_REGISTERS:
        return list(regs.GP_REGISTERS[width]), "REG", bool(src), bool(dest), width, signed

    # ---- SIMD / FP register ----
    if "SIMD" in text and width in regs.SIMD_REGISTERS:
        return list(regs.SIMD_REGISTERS[width]), "REG", bool(src), bool(dest), width, signed

    # ---- ARM bitmask immediate ----
    if "bitmask" in text:
        if data_size_hint is None:
            raise ParseFailed(
                f"bitmask immediate for {variant_name}:{variable} "
                "but no preceding register operand to infer data size"
            )
        values = generate_logical_immediates(bits=data_size_hint)
        return values, "IMM", True, False, data_size_hint, False

    # ---- SVE scalable vector ----
    if "scalable vector" in text:
        return list(regs.SVE_VECTOR_REGISTERS), "REG", bool(src), bool(dest), 0, False

    # ---- SVE predicate ----
    if "scalable predicate" in text:
        return list(regs.SVE_PREDICATE_REGISTERS), "REG", bool(src), bool(dest), 0, False

    # ---- condition code (hint already set by _handle_table, or explicit) ----
    if hint == "COND":
        return list(regs.CONDITION_CODES), "COND", False, False, 0, False

    # ---- numeric range expressed as "the number in the range [a-b]" ----
    m_bracket = re.search(r"\[([+-]?\d+)-([+-]?\d+)\]", text)
    if "the number" in text and m_bracket:
        a, b = int(m_bracket.group(1)), int(m_bracket.group(2))
        signed = a < 0
        width = max(1, int(log2(b - a + 1))) if b > a else 1
        return [f"[{a}-{b}]"], "IMM", True, False, width, signed

    # ---- immediate with explicit "X to Y" range ----
    if hint == "IMM":
        m_range = re.search(r"([+-]?\d+) to ([+-]?\d+)", text)
        if not m_range:
            raise ParseFailed(
                f"IMM hint but no numeric range found for {variant_name}:{variable} "
                f"(text: {text[:120]!r})"
            )
        a, b = int(m_range.group(1)), int(m_range.group(2))
        signed = a < 0
        width = max(1, int(log2(b - a + 1))) if b > a else 1
        m_mult = re.search(r"multiple of ([+]?\d+)", text)
        step = int(m_mult.group(1)) if m_mult else 1
        values = [str(v) for v in range(a, b + 1) if v % step == 0]
        return values, "IMM", True, False, width, "unsigned" not in text

    raise ParseFailed(
        f"Cannot determine operand kind for {variant_name}:{variable} "
        f"(hint={hint!r}, text={text[:120]!r})"
    )


# ---------------------------------------------------------------------------
# Register-width split
# ---------------------------------------------------------------------------
#
# Some instructions encode register width as part of the instruction encoding
# itself rather than as a fixed property of the mnemonic — the programmer
# chooses Wn (32-bit) or Xn (64-bit) at the call site, and that choice
# structurally constrains other operands (most notably the bit-index immediate
# in TBZ/TBNZ, where the 6-bit encoding field uses bit 5 as both the MSB of
# the index and the register-width selector).
#
# If the XML emits a single encoding covering both widths (i.e. the register
# operand's values list contains both W and X registers), we split it into two
# InstructionSpec objects — one per width — and apply the correct constraints
# to each.
#
# _WIDTH_SPLIT_MNEMONICS: the set of mnemonics that need this treatment.
# _IMM_RANGES_BY_WIDTH: for each mnemonic and register width, the allowed
#   [lo, hi] for every IMM operand.  A None entry means "no constraint"
#   (leave the values list untouched).
# ---------------------------------------------------------------------------

_WIDTH_SPLIT_MNEMONICS: frozenset[str] = frozenset({"tbz", "tbnz"})

# Maps mnemonic → { reg_width → (lo, hi) } for the IMM bit-index operand.
# Any mnemonic not in this dict but in _WIDTH_SPLIT_MNEMONICS gets split by
# width with no additional IMM filtering (future-proof placeholder).
_IMM_RANGE_BY_WIDTH: dict[str, dict[int, tuple[int, int]]] = {
    "tbz":  {32: (0, 31), 64: (0, 63)},
    "tbnz": {32: (0, 31), 64: (0, 63)},
}


def _split_by_register_width(spec: InstructionSpec) -> list[InstructionSpec]:
    """
    Split *spec* into one InstructionSpec per distinct register width found
    in its REG operands, then apply the per-width IMM constraints defined in
    _IMM_RANGE_BY_WIDTH.

    If the spec's register operands are already pure (all one width), the
    original spec is returned unchanged in a one-element list — no copy is
    made.

    Steps for each width bucket:
      1. Deep-copy the spec.
      2. Filter every REG operand to only the registers of that width.
      3. Filter every IMM operand to the allowed range for that width.
      4. Drop the copy if any REG or IMM operand ends up empty.
    """
    gp32 = set(regs.GP_REGISTERS[32])
    gp64 = set(regs.GP_REGISTERS[64])

    # Determine which widths are actually present in the spec's REG operands.
    has_w = any(
        v in gp32
        for op in spec.operands if op.type_ == "REG"
        for v in op.values
    )
    has_x = any(
        v in gp64
        for op in spec.operands if op.type_ == "REG"
        for v in op.values
    )

    # Already pure — nothing to split.
    if has_w == has_x == False:
        return [spec]
    if has_w and not has_x:
        _apply_imm_range(spec, spec.name.lower(), 32)
        return [spec]
    if has_x and not has_w:
        _apply_imm_range(spec, spec.name.lower(), 64)
        return [spec]

    # Mixed: produce one copy per width.
    results: list[InstructionSpec] = []
    for width, keep_set in ((32, gp32), (64, gp64)):
        s = copy.deepcopy(spec)
        valid = True

        for op in s.operands:
            if op.type_ != "REG":
                continue
            op.values = [v for v in op.values if v in keep_set]
            if not op.values:
                valid = False
                break

        if not valid:
            continue

        _apply_imm_range(s, s.name.lower(), width)

        # Drop if any IMM operand ended up empty after filtering.
        if any(op.type_ == "IMM" and not op.values for op in s.operands):
            continue

        results.append(s)

    # Fallback: if splitting produced nothing, return the original.
    return results if results else [spec]


def _apply_imm_range(spec: InstructionSpec, mnemonic: str, reg_width: int) -> None:
    """
    Apply the per-width IMM constraint from _IMM_RANGE_BY_WIDTH to *spec*
    in-place.  No-ops if the mnemonic has no entry in the table.
    """
    width_map = _IMM_RANGE_BY_WIDTH.get(mnemonic)
    if width_map is None:
        return
    lo, hi = width_map.get(reg_width, (None, None))
    if lo is None:
        return
    for op in spec.operands:
        if op.type_ == "IMM":
            _filter_imm_values(op, lo, hi)


def _postprocess(spec: InstructionSpec) -> bool:
    """
    Return False for encodings that should be dropped.

    Rules (matching the original implementation):
    - Drop any encoding that uses an 'extend' modifier.
    - Drop if LSL appears in extend but there is no 'amount' operand.
    - Drop memory encodings where a 64-bit base reg precedes a 32-bit offset
      reg without an extend clause (ambiguous addressing).
    - Drop non-control-flow instructions that have a LABEL operand.
    - Drop encodings that violate AArch64 implicit immediate-range constraints
      (see _validate_encoding_constraints for full rule list).
    """
    has_extend = False
    has_lsl_in_extend = False
    has_amount = False
    mem_op_32: Optional[OperandSpec] = None
    mem_op_64: Optional[OperandSpec] = None

    gp32 = set(regs.GP_REGISTERS[32])
    gp64 = set(regs.GP_REGISTERS[64])

    for op in spec.operands:
        if op.name == "extend":
            has_extend = True
            has_lsl_in_extend = any(v.lower() == "lsl" for v in op.values)
        elif op.name == "amount":
            has_amount = True
        elif op.type_ == "MEM":
            if any(v in gp32 for v in op.values):
                mem_op_32 = op
            if any(v in gp64 for v in op.values):
                mem_op_64 = op
        elif op.type_ == "LABEL" and not spec.control_flow:
            return False

    if has_extend:
        return False

    if has_lsl_in_extend and not has_amount:
        return False

    if mem_op_64 and mem_op_32 and not has_extend:
        # If the 64-bit base appears before the 32-bit offset in the template,
        # an extend clause would be required — drop this encoding.
        if spec.template.find(mem_op_64.name) < spec.template.find(mem_op_32.name):
            return False

    reason = _validate_encoding_constraints(spec)
    if reason:
        logger.debug("Dropped %s: %s", spec.name, reason)
        return False

    return True


# ---------------------------------------------------------------------------
# AArch64 implicit encoding-constraint validation
# ---------------------------------------------------------------------------
#
# These rules encode constraints that the ARM ISA XML either documents poorly
# or only implies through the encoding structure.  Each check returns a short
# diagnostic string on failure, or None on success.  The dispatcher
# _validate_encoding_constraints tries every applicable rule for the given
# mnemonic and returns the first failure reason (or None if all pass).
#
# Rule catalogue:
#
#  TBZ / TBNZ
#      The bit-index immediate is encoded as a 6-bit field where b5 doubles as
#      the register-width selector.  When a 32-bit Wn register is used, b5 must
#      be 0, so the bit index is limited to [0, 31].
#
#  UBFM / SBFM / BFM  (and aliases LSL/LSR/ASR/ROR/UBFIZ/SBFIZ/BFI/BFXIL)
#      The N bit in the encoding selects 32- vs 64-bit operation.  When N=0
#      (32-bit form), immr and imms must each be < 32.  Values >= 32 with N=0
#      are UNDEFINED behaviour.
#
#  EXTR  (and its ROR alias)
#      Same N/immr/imms constraint as the BFM family.
#
#  LSL / LSR / ASR / ROR  immediate forms (aliases of UBFM / SBFM / EXTR)
#      Shift amount must be in [0, 31] for 32-bit registers, [0, 63] for 64-bit.
#      This is derived from the BFM-family constraint above; it is listed
#      separately because these aliases expose the shift as a standalone IMM
#      operand whose values list must be filtered accordingly.
#
#  LSLV / LSRV / ASRV / RORV  (register-shift forms)
#      The shift register value is used modulo 32 or 64 — upper bits are
#      silently ignored by hardware.  No values need to be dropped, but the
#      width annotation on the shift-register operand must match the data
#      register width; mixed-width pairs are UNDEFINED.
#
#  CCMP / CCMN
#      The nzcv immediate is a 4-bit field: values must be in [0, 15].
#      The condition-code field has 16 encodings but NV (1111b = "nv") is
#      UNPREDICTABLE in CCMP/CCMN; it is removed from the COND operand values.
#
#  LDR / STR  (unsigned-offset form)
#      The 12-bit immediate is *scaled* by the access size.  This means the
#      raw byte offset encoded must be a multiple of the access size, and the
#      maximum byte offset is (4095 * access_size).  The parser's values list
#      (populated from the XML "range" annotation) is filtered to multiples of
#      the correct stride.
#
#  LDUR / STUR  (unscaled-offset form)
#      9-bit *signed* immediate in [-256, 255].  Verify the values list
#      contains no out-of-range entries (the XML sometimes annotates these
#      with the wrong sign or range).
#
#  MRS / MSR  (system-register access)
#      Many op0:op1:CRn:CRm:op2 combinations are UNDEFINED or RES0.
#      We do not enumerate the full table here; instead we flag encodings
#      where the immediate operand (the encoded sysreg index) has no
#      values at all — a sign the XML described it as a free-range field
#      without listing the valid entries.
#
#  HINT
#      7-bit immediate in [0, 127].  Many values are NOP-equivalent
#      architecturally, but some are defined (e.g. 0=NOP, 1=YIELD,
#      20=BTI).  Values outside [0, 127] are filtered out.
# ---------------------------------------------------------------------------

# Mnemonics that belong to the BFM / bitfield family
_BITFIELD_MNEMONICS: frozenset[str] = frozenset({
    "ubfm", "sbfm", "bfm",
    # Common aliases
    "lsl", "lsr", "asr", "ror",
    "ubfiz", "ubfx", "sbfiz", "sbfx",
    "bfi", "bfxil", "bfc",
    # Rotate-extract (EXTR) and its ROR alias
    "extr",
})

# Mnemonics for the register-shift variants
_REG_SHIFT_MNEMONICS: frozenset[str] = frozenset({
    "lslv", "lsrv", "asrv", "rorv",
})

# Load/store unsigned-offset (scaled) mnemonics → their natural access size
# (bytes).  This is the stride used to scale the 12-bit pimm field.
_SCALED_LS_ACCESS_SIZE: dict[str, int] = {
    # 1-byte
    "strb": 1, "ldrb": 1, "ldrsb": 1,
    # 2-byte
    "strh": 2, "ldrh": 2, "ldrsh": 2,
    # 4-byte
    "str":  4, "ldr":  4, "ldrsw": 4,
    # 8-byte
    "strd": 8, "ldrd": 8,
    # SIMD scalar variants
    "str_s": 4, "ldr_s": 4,
    "str_d": 8, "ldr_d": 8,
    "str_q": 16, "ldr_q": 16,
}

# Mnemonics for the unscaled-offset (LDUR/STUR) family
_UNSCALED_LS_MNEMONICS: frozenset[str] = frozenset({
    "ldur", "stur", "ldurb", "sturb", "ldurh", "sturh",
    "ldursb", "ldursh", "ldursw",
    "ldurq", "sturq", "ldurd", "sturd", "ldurs", "sturs",
})


def _reg_width_from_operands(spec: InstructionSpec) -> Optional[int]:
    """
    Return the data-register width (32 or 64) inferred from the first GP
    register operand, or None if it cannot be determined.
    """
    gp32 = set(regs.GP_REGISTERS[32])
    gp64 = set(regs.GP_REGISTERS[64])
    for op in spec.operands:
        if op.type_ == "REG":
            if any(v in gp32 for v in op.values):
                return 32
            if any(v in gp64 for v in op.values):
                return 64
    return None


def _filter_imm_values(op: OperandSpec, lo: int, hi: int) -> None:
    """
    Remove values outside [lo, hi] from an IMM operand's values list in-place.
    Handles both plain integers and range-notation strings like "[0-31]".
    """
    kept: list[str] = []
    for v in op.values:
        # Range notation "[a-b]"
        m = re.fullmatch(r"\[([+-]?\d+)-([+-]?\d+)\]", v)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            # Clamp the range to [lo, hi]
            a2, b2 = max(a, lo), min(b, hi)
            if a2 <= b2:
                kept.append(f"[{a2}-{b2}]")
        else:
            try:
                if lo <= int(v) <= hi:
                    kept.append(v)
            except ValueError:
                kept.append(v)   # non-numeric — leave it alone
    op.values = kept


def _check_bitfield_family(spec: InstructionSpec) -> Optional[str]:
    """
    UBFM/SBFM/BFM and aliases: for the 32-bit form (N=0), immr and imms
    must each be < 32.  Values >= 32 with N=0 are UNDEFINED.

    We identify "32-bit form" by the presence of W-registers, and clamp
    every IMM operand's values to [0, 31].
    """
    reg_width = _reg_width_from_operands(spec)
    if reg_width != 32:
        return None   # 64-bit form allows the full [0, 63] range

    for op in spec.operands:
        if op.type_ != "IMM":
            continue
        before = list(op.values)
        _filter_imm_values(op, 0, 31)
        if not op.values:
            return (
                f"{spec.name.upper()} 32-bit form: IMM operand '{op.name}' has "
                f"no valid values after restricting to [0, 31] (was: {before})"
            )
    return None


def _check_shift_immediate(spec: InstructionSpec) -> Optional[str]:
    """
    LSL/LSR/ASR/ROR immediate forms: shift amount in [0, 31] for 32-bit
    registers, [0, 63] for 64-bit.  These are aliases of UBFM/SBFM/EXTR
    so the constraint is the same, but it surfaces through a standalone
    shift-amount IMM operand rather than separate immr/imms operands.
    """
    reg_width = _reg_width_from_operands(spec)
    if reg_width is None:
        return None
    hi = reg_width - 1   # 31 or 63

    for op in spec.operands:
        if op.type_ != "IMM":
            continue
        before = list(op.values)
        _filter_imm_values(op, 0, hi)
        if not op.values:
            return (
                f"{spec.name.upper()} {reg_width}-bit form: shift-amount "
                f"immediate '{op.name}' has no valid values after restricting "
                f"to [0, {hi}] (was: {before})"
            )
    return None


def _check_reg_shift_width(spec: InstructionSpec) -> Optional[str]:
    """
    LSLV/LSRV/ASRV/RORV: the shift register and the data register must have
    the same width.  A mixed-width pair (e.g. X0, W1) is UNDEFINED.
    Returns a failure reason if the operand widths are mismatched; otherwise
    filters any cross-width register values from shared operands.
    """
    gp32 = set(regs.GP_REGISTERS[32])
    gp64 = set(regs.GP_REGISTERS[64])
    widths: list[int] = []
    for op in spec.operands:
        if op.type_ != "REG":
            continue
        if any(v in gp32 for v in op.values):
            widths.append(32)
        elif any(v in gp64 for v in op.values):
            widths.append(64)

    if len(set(widths)) > 1:
        return (
            f"{spec.name.upper()}: mixed-width register operands "
            f"{widths} — UNDEFINED in AArch64"
        )
    return None


def _check_ccmp_ccmn(spec: InstructionSpec) -> Optional[str]:
    """
    CCMP/CCMN:
      - nzcv immediate must be in [0, 15] (4-bit field).
      - Condition code 'nv' (encoding 0b1111) is UNPREDICTABLE; remove it.
    """
    for op in spec.operands:
        if op.type_ == "IMM":
            before = list(op.values)
            _filter_imm_values(op, 0, 15)
            if not op.values:
                return (
                    f"{spec.name.upper()}: nzcv immediate has no valid values "
                    f"after restricting to [0, 15] (was: {before})"
                )
        elif op.type_ == "COND":
            # Remove the UNPREDICTABLE 'nv' condition code
            op.values = [v for v in op.values if v.lower() != "nv"]
            if not op.values:
                return f"{spec.name.upper()}: all condition codes removed"
    return None


def _check_ldr_str_scaled(spec: InstructionSpec, access_size: int) -> Optional[str]:
    """
    LDR/STR unsigned-scaled-offset form: the 12-bit pimm field is scaled by
    *access_size* bytes.  Valid byte offsets are multiples of access_size in
    [0, 4095 * access_size].  Filter the IMM values list to only those
    multiples; plain integers that are not multiples are removed.
    """
    max_offset = 4095 * access_size
    for op in spec.operands:
        if op.type_ != "IMM":
            continue
        kept: list[str] = []
        for v in op.values:
            m = re.fullmatch(r"\[([+-]?\d+)-([+-]?\d+)\]", v)
            if m:
                a, b = int(m.group(1)), int(m.group(2))
                # Align range bounds to stride multiples
                a2 = ((max(a, 0) + access_size - 1) // access_size) * access_size
                b2 = (min(b, max_offset) // access_size) * access_size
                if a2 <= b2:
                    kept.append(f"[{a2}-{b2}]")
            else:
                try:
                    iv = int(v)
                    if 0 <= iv <= max_offset and iv % access_size == 0:
                        kept.append(v)
                except ValueError:
                    kept.append(v)
        if not kept:
            return (
                f"{spec.name.upper()}: scaled-offset immediate has no valid "
                f"multiples of {access_size} in [0, {max_offset}]"
            )
        op.values = kept
    return None


def _check_ldur_stur(spec: InstructionSpec) -> Optional[str]:
    """
    LDUR/STUR unscaled-offset form: 9-bit signed immediate in [-256, 255].
    Filter values list to that range.
    """
    for op in spec.operands:
        if op.type_ != "IMM":
            continue
        before = list(op.values)
        _filter_imm_values(op, -256, 255)
        if not op.values:
            return (
                f"{spec.name.upper()}: unscaled-offset immediate has no valid "
                f"values in [-256, 255] (was: {before})"
            )
    return None


def _check_mrs_msr(spec: InstructionSpec) -> Optional[str]:
    """
    MRS/MSR: flag encodings where the system-register IMM operand has no
    values — the XML described it as a free-range field without enumerating
    valid op0:op1:CRn:CRm:op2 combinations.
    """
    for op in spec.operands:
        if op.type_ == "IMM" and not op.values:
            return (
                f"{spec.name.upper()}: system-register operand '{op.name}' has "
                f"no enumerated values — encoding would generate UNDEFINED sysreg accesses"
            )
    return None


def _check_hint(spec: InstructionSpec) -> Optional[str]:
    """
    HINT: 7-bit immediate in [0, 127].  Values outside this range are not
    encodable; filter them out.
    """
    for op in spec.operands:
        if op.type_ != "IMM":
            continue
        before = list(op.values)
        _filter_imm_values(op, 0, 127)
        if not op.values:
            return (
                f"HINT: immediate has no valid values in [0, 127] "
                f"(was: {before})"
            )
    return None


# Dispatch table: mnemonic (lower-case) → checker function
# A checker receives the full InstructionSpec and returns Optional[str]:
#   None   → constraint satisfied (keep the encoding)
#   str    → constraint violated (drop; string is the log reason)
_CONSTRAINT_CHECKS: dict[str, callable] = {
    # TBZ/TBNZ: handled entirely by _split_by_register_width before _postprocess;
    # no residual check needed here because each split spec is already pure-width.
    **{m: _check_bitfield_family for m in _BITFIELD_MNEMONICS},
    **{m: _check_reg_shift_width for m in _REG_SHIFT_MNEMONICS},
    "ccmp": _check_ccmp_ccmn,
    "ccmn": _check_ccmp_ccmn,
    "hint": _check_hint,
    "mrs":  _check_mrs_msr,
    "msr":  _check_mrs_msr,
    **{m: _check_ldur_stur for m in _UNSCALED_LS_MNEMONICS},
}

# Shift-immediate aliases share the same check but need their own entries
# so they don't collide with the bitfield-family entry for "lsl" etc.
# (The bitfield check and the shift-immediate check are compatible — the
#  bitfield check clamps immr/imms while the shift check clamps the shift
#  amount; for the alias forms the single IMM operand is the shift amount.)
_SHIFT_IMM_ALIASES: frozenset[str] = frozenset({"lsl", "lsr", "asr", "ror"})


def _validate_encoding_constraints(spec: InstructionSpec) -> Optional[str]:
    """
    Run all applicable constraint checks for *spec*.

    Returns the first failure reason string, or None if all checks pass.
    Values lists on operands may be mutated in-place by the checkers even
    when None is returned (i.e., filtering is applied even on kept specs).
    """
    name = spec.name.lower()

    # Scaled load/store: look up by mnemonic for access size, then check
    if name in _SCALED_LS_ACCESS_SIZE:
        reason = _check_ldr_str_scaled(spec, _SCALED_LS_ACCESS_SIZE[name])
        if reason:
            return reason

    # Shift-immediate aliases: apply shift-amount range check *in addition to*
    # any bitfield-family check (they are redundant for the alias forms but
    # harmless and serve as a belt-and-suspenders defence).
    if name in _SHIFT_IMM_ALIASES:
        reason = _check_shift_immediate(spec)
        if reason:
            return reason

    # General dispatch
    checker = _CONSTRAINT_CHECKS.get(name)
    if checker:
        return checker(spec)

    return None


def _extract_flags_operand(section: ET.Element) -> Optional[OperandSpec]:
    """
    Scan the ASL pseudo-code in *section* to determine which NZCV flags are
    read and/or written, and return a FLAGS OperandSpec (or None).

    The values list follows the x86 EFLAGS slot order used by Revizor:
    [C, _, _, Z, N, _, _, _, V]
    """
    pstext = section.find("ps_section/ps/pstext")
    if pstext is None:
        return None

    # flag → [is_read, is_written]
    flag_state: dict[str, list[bool]] = {f: [False, False] for f in "NZCV"}
    flag_state[""] = [False, False]   # placeholder for unused slots
    uses_flags = False

    pstate_pattern = re.compile(r"\bPSTATE\.(?:<|\[)?([NZCV](?:,[NZCV])*)(?:>|\])?")

    cond_pattern = re.compile(r"\bConditionHolds\s*\(")

    text = "".join(pstext.itertext())

    for m in pstate_pattern.finditer(text):
        flags = [f.strip() for f in m.group(1).split(",")]

        # Local context around match
        before = text[max(0, m.start() - 4):m.start()]
        after  = text[m.end():m.end() + 4]

        # Detect write: PSTATE... =
        is_write = bool(re.match(r"\s*=", after))

        # Detect read: = PSTATE...
        is_read = bool(re.search(r"=\s*$", before))

        # If neither → it's usage in expression → READ
        if not is_read and not is_write:
            is_read = True

        uses_flags = True

        for f in flags:
            if f in flag_state:
                if is_read:
                    flag_state[f][0] = True
                if is_write:
                    flag_state[f][1] = True


    # --- ConditionHolds → implicit read of flags ---
    if cond_pattern.search(text):
        uses_flags = True
        for f in ["N", "Z", "C", "V"]:
            if f in flag_state:
                flag_state[f][0] = True


    if not uses_flags:
        return None

    # Map to the 9-slot Revizor order: C _ _ Z N _ _ _ V
    slots = ["C", "", "", "Z", "N", "", "", "", "V"]
    values: list[str] = []
    is_src = False
    is_dest = False
    for f in slots:
        r, w = flag_state.get(f, [False, False])
        # A read flag makes the operand a source; a written flag makes it a destination.
        if r:
            is_src = True
        if w:
            is_dest = True

        if r and w:
            values.append("r/w")
        elif r:
            values.append("r")
        elif w:
            values.append("w")
        else:
            values.append("")

    assert len(values) == len(slots)
    return OperandSpec("flags", "FLAGS", values, src=is_src, dest=is_dest, width=0, signed=False)

