"""
File: Parsing of assembly files into our internal representation (TestCase).
      AArch64-specific.
"""
import re
import os
import weakref
from typing import List, Dict

from .aarch64_generator import Aarch64Generator
from ..asm_parser import AsmParserGeneric, parser_assert, parser_error
from ..interfaces import OT, Instruction, InstructionSpec, LabelOperand, RegisterOperand, \
    MemoryOperand, MemorySpec, ImmediateOperand, CondOperand


_TEMPLATE_REGEX_CACHE = weakref.WeakKeyDictionary()


def _flatten_specs(spec):
    """The operand specs that own a template placeholder, in order: a memory access contributes its
    address components (the `[...]` placeholders), every other operand contributes itself. Mirrors
    Instruction.to_asm_string, which substitutes the inner components by name."""
    flat = []
    for op in spec.operands:
        flat.extend(op.inner) if isinstance(op, MemorySpec) else flat.append(op)
    return flat

def _operand_pattern(op) -> str:
    """Regex fragment for one operand. If the operand has a small enumerable value set we match
    those exactly (this also distinguishes 32-bit w-regs from 64-bit x-regs straight from the
    data); otherwise fall back to a type-based pattern."""
    vals = list(op.values) if op.values else []
    if vals and len(vals) <= 64:
        return "(?:" + "|".join(re.escape(v) for v in sorted(vals, key=len, reverse=True)) + ")"
    if op.type == OT.IMM:
        return r"-?(?:0x[0-9a-fA-F]+|0b[01]+|[0-9]+)"
    if op.type == OT.LABEL:
        return r"\.[^\s,]+"
    return r"[^\s,\[\]#!]+"

def _template_regex(spec):
    """Compile (and cache) an anchored regex from a spec's template: literal text (mnemonic,
    brackets, '#', ',', '!') is matched verbatim, each {placeholder} becomes a capture group
    built from the matching operand, in template/operand order. Returns None if the template's
    placeholder count does not line up with the operand count (so it is simply never matched)."""
    rx = _TEMPLATE_REGEX_CACHE.get(spec, False)
    if rx is not False:
        return rx
    flat = _flatten_specs(spec)            # one placeholder per flattened operand (memory -> its parts)
    parts, op_idx = ["^"], 0
    for tok in re.split(r"(\{[^}]*\}|\s+)", (spec.template or "").strip()):
        if not tok:
            continue
        if tok.isspace():
            parts.append(r"\s*")
        elif tok[0] == "{" and tok[-1] == "}":
            if op_idx >= len(flat):
                op_idx = -1
                break
            parts.append("(" + _operand_pattern(flat[op_idx]) + ")")
            op_idx += 1
        else:
            parts.append(re.escape(tok))
    rx = None
    if op_idx == len(flat) and spec.template:
        parts.append("$")
        rx = re.compile("".join(parts), re.IGNORECASE)
    _TEMPLATE_REGEX_CACHE[spec] = rx
    return rx


class Aarch64AsmParser(AsmParserGeneric):
    generator: Aarch64Generator

    asm_prefixes = []
    asm_synonyms = {}
    memory_sizes = {}

    def parse_line(self, line: str, line_num: int,
                   instruction_map: Dict[str, List[InstructionSpec]]) -> Instruction:
        raw_line = line
        line = line.strip().lower()
        is_instrumentation = "instrumentation" in line
        is_noremove = "noremove" in line

        m = re.match(r"^\s*(?P<mnemonic>[a-z][a-z0-9.]*)(?:\s+(?P<operands>[^/]+?))?\s*(?://.*)?$",
                     line)
        parser_assert(m is not None, line_num, f"Failure to parse instruction: {raw_line}")
        name = m.group("mnemonic")
        operands_raw = (m.group("operands") or "").strip()

        # Synthetic pseudo-instructions (no real asm template): parse their operands directly.
        if name in ("opcode", "macro"):
            return self._build_instruction(instruction_map[name][0],
                                            [t.strip() for t in operands_raw.split(",") if t.strip()],
                                            is_instrumentation, is_noremove, line_num)

        # Real instruction: match the line against every candidate spec's template.
        lookup = name
        if name not in instruction_map and "." in name:
            lookup = name.split(".", 1)[0] + "."     # b.eq -> b.
        candidates = instruction_map.get(lookup, [])
        parser_assert(len(candidates) > 0, line_num, f"Unknown instruction {line}")

        code = re.sub(r"\s+", " ", name if not operands_raw else f"{name} {operands_raw}")
        matches = []
        for spec in candidates:
            rx = _template_regex(spec)
            if rx is None:
                continue
            mm = rx.match(code)
            if mm:
                matches.append((spec, list(mm.groups())))

        # Tie-break on magic-value specs (as before), then require a single unambiguous match.
        if len(matches) > 1:
            magic = [(s, v) for s, v in matches if s.has_magic_value]
            if magic:
                matches = magic
        parser_assert(len(matches) > 0, line_num, f"Could not find matching spec for {line}")
        # Only a genuine ambiguity is an error: specs that would build a *different* instruction
        # (distinct operand shape). Specs differing only cosmetically (e.g. {Wn|WSP} vs {Wn})
        # yield the same parse, so picking the first is safe.
        shapes = {tuple((o.type, o.width) for o in s.operands) for s, _ in matches}
        parser_assert(len(shapes) == 1, line_num,
                      f"Ambiguous parse: {len(shapes)} distinct operand shapes match {line!r}")

        spec, values = matches[0]
        return self._build_instruction(spec, values, is_instrumentation, is_noremove, line_num)

    def _simple_operand(self, op_spec, value, line_num):
        if op_spec.type == OT.REG:
            op = RegisterOperand(value, op_spec.width, op_spec.src, op_spec.dest)
        elif op_spec.type == OT.IMM:
            op = ImmediateOperand(value, op_spec.width)
        elif op_spec.type == OT.LABEL:
            op = LabelOperand(value)
        elif op_spec.type == OT.COND:
            op = CondOperand(value)
        else:
            parser_error(line_num, f"Unknown operand type {op_spec.type}")
        op.name = op_spec.name
        op.mem_role = op_spec.mem_role          # address components carry their role for the sandbox
        return op

    def _build_instruction(self, spec, values, is_instrumentation, is_noremove, line_num):
        inst = Instruction.from_spec(spec, is_instrumentation=is_instrumentation)
        inst.is_noremove = is_noremove
        vi = 0
        for op_spec in spec.operands:
            if isinstance(op_spec, MemorySpec):
                # consume one captured value per inner component and wrap them as a memory access
                inner = [self._simple_operand(c, values[vi + k], line_num)
                         for k, c in enumerate(op_spec.inner)]
                vi += len(op_spec.inner)
                address = ", ".join(o.value for o in inner)
                op = MemoryOperand(address, op_spec.width, op_spec.src, op_spec.dest, inner=inner)
                op.name = op_spec.name
            else:
                op = self._simple_operand(op_spec, values[vi], line_num)
                vi += 1
            inst.operands.append(op)
        for op_spec in spec.implicit_operands:
            inst.implicit_operands.append(self.generator.generate_operand(op_spec, inst))
        return inst

    def _patch_asm(self, asm_file: str, patched_asm_file: str):
        """
        Ensure function labels are exposed in AArch64:
          - Adds a global function label if missing.
          - Adds NOP at function end for easier size calculations.
          - Inserts `.function_0` if missing.
          - Ensures `.test_case_exit` is within `.data.main` //with a single NOP.
        """

        def is_instruction(line: str) -> bool:
            """Check if the line is an actual instruction or data directive."""
            return line and not line.startswith("#") and (
                not line.startswith(".") or
                line.startswith((
                                ".bcd", ".byte", ".long", ".quad", ".macro", ".value", ".2byte",
                                ".4byte", ".8byte"))
            )

        main_function_label = ""
        enter_found = False
        has_measurement_start = False
        has_measurement_end = False
        prev_line = ""

        with open(asm_file, "r") as f:
            with open(patched_asm_file, "w") as patched:
                for line in f:
                    line = line.strip().lower()

                    if line.startswith(".macro.measurement_start"):
                        has_measurement_start = True
                    elif line.startswith(".macro.measurement_end"):
                        has_measurement_end = True

                    if not enter_found:
                        if line == ".test_case_enter:":
                            enter_found = True
                        patched.write(line + "\n")
                        continue

                    if ".test_case_exit:" in line:
                        if not main_function_label:
                            patched.write(".function_0:\n")
                            main_function_label = ".function_0"
                        if ".data.main" not in prev_line or "measurement_end" in prev_line:
                            patched.write(".section .data.main\n")
                        patched.write(".test_case_exit:\n")
                        continue

                    if line.startswith(".function_") and not main_function_label:
                        main_function_label = line[:-1]
                    elif not main_function_label and is_instruction(line):
                        patched.write(".function_0:\n")
                        main_function_label = ".function_0"

                    patched.write(line + "\n")
                    prev_line = line

        macro_placeholder = " nop"  # AArch64 NOP (0xd503201f)

        # Add jump placeholders after macros
        with open(patched_asm_file, "r") as f:
            with open(patched_asm_file + ".tmp", "w") as patched:
                for line in f:
                    line = line.lower()
                    if line.startswith(".macro"):
                        if "nop" not in line:
                            patched.write(line[:-1] + macro_placeholder + "\n")
                        else:
                            if macro_placeholder not in line:
                                raise RuntimeError("Unexpected NOP placeholder: " + line)
                            patched.write(line)
                    else:
                        patched.write(line)
        os.rename(patched_asm_file + ".tmp", patched_asm_file)

        # Add .macro.measurement_start after .function_0
        if not has_measurement_start:
            with open(patched_asm_file, "r") as f:
                with open(patched_asm_file + ".tmp", "w") as patched:
                    for line in f:
                        line = line.lower()
                        patched.write(line)
                        if line.startswith(main_function_label):
                            patched.write(".macro.measurement_start:\n    nop\n")
            os.rename(patched_asm_file + ".tmp", patched_asm_file)

        # Add .macro.measurement_end before .test_case_exit
        if not has_measurement_end:
            with open(patched_asm_file, "r") as f:
                with open(patched_asm_file + ".tmp", "w") as patched:
                    prev_line = ""
                    for line in f:
                        line = line.lower()
                        if line.startswith(".test_case_exit:"):
                            if prev_line.startswith(".section"):
                                patched.write(".function_end:\n")
                            patched.write(".macro.measurement_end:\n    nop\n")
                        patched.write(line)
                        prev_line = line
            os.rename(patched_asm_file + ".tmp", patched_asm_file)
