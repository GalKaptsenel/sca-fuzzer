from __future__ import annotations
import re
import xml.etree.ElementTree as ET
from .models import Instruction, FlagEffects, EncodingCtx, ExtractionError, OperandKind
from .asl import extract_asl_semantics
from .operands import build_operands
from .immediate import constant_fields
from .constraints import unpredictable_constraints
from .validate import validate_instruction

_COMMENT = re.compile(r"//[^\n]*|/\*.*?\*/", re.S)  # ASL comments must not contribute reads/writes


def _section_asl(section: ET.Element, want: str) -> str:
    text = "\n".join("".join(t.itertext()) for t in section.iter("pstext")
                     if t.get("section") == want)
    return _COMMENT.sub("", text)


def _box_width(box: ET.Element) -> int:
    w = box.get("width")            # ARM DTD: a box with no width attribute is a single bit
    return int(w) if w is not None else 1


def _docvar(el: ET.Element, key: str) -> str | None:
    dv = el.find("docvars")
    if dv is None:
        return None
    for d in dv.iter("docvar"):
        if d.get("key") == key:
            return d.get("value")
    return None


def _build_instruction(section_id, category, explanations, asl, postdecode, sem, control_flow,
                       flags, arch_constants, decode, boxes, rd, enc, enc_name) -> Instruction:
    tmpl = enc.find("asmtemplate")
    text = tmpl.find("text") if tmpl is not None else None
    if text is None or not text.text:
        raise ExtractionError(f"{enc_name}: no asmtemplate text")
    asm = "".join(tmpl.itertext())
    ctx = EncodingCtx(boxes, constant_fields(rd, enc), arch_constants, decode, asl)
    operands = build_operands(enc_name, asm, explanations, sem, ctx)
    reg_vars = frozenset(o.asl_index for o in operands if o.kind is OperandKind.REG and o.asl_index)
    # the mnemonic is the leading run of mnemonic chars in the first asm-text node, i.e. before any
    # `<...>` sub-token or optional `{...}` syntax: `SQDMULL{` -> sqdmull, `B.` (+<cond>) -> b.
    name_match = re.match(r"[A-Za-z0-9.]+", text.text.split()[0])
    if name_match is None:
        raise ExtractionError(f"{enc_name}: cannot read mnemonic from {text.text!r}")
    name = name_match.group(0).lower()
    inst = Instruction(
        name=name, iclass_id=section_id, category=category,
        encoding_name=enc_name, asm_template=asm, control_flow=control_flow,
        mem_access=sem.mem_access, flags=flags, operands=operands,
        constraints=unpredictable_constraints(decode + "\n" + postdecode, reg_vars))
    validate_instruction(inst)   # type/domain check; a malformed field loud-fails this encoding
    return inst


def iter_instructions(path, arch_constants: dict, failures: dict):
    """Yield one Instruction per encoding. Every encoding (or section) we cannot handle records
    `failures[name] = reason` and is skipped, so one unhandled encoding never hides its siblings —
    a miss is always reported, never silently dropped."""
    tree = ET.parse(path)
    for section in tree.iter("instructionsection"):
        if section.get("type") != "instruction":
            continue
        section_id = section.get("id")
        if section_id is None:
            failures[path] = "instructionsection has no id"
            continue
        try:
            if not any(t.get("section") == "Execute" for t in section.iter("pstext")):
                raise ExtractionError(f"{section_id}: no Execute ASL")
            asl = _section_asl(section, "Execute")  # Execute is section-level (shared semantics)
            postdecode = _section_asl(section, "Postdecode")  # shared; holds LDP/LDR overlap guards
            sem = extract_asl_semantics(asl)
            control_flow = ("BranchTo" in asl) or ("BranchNotTaken" in asl)
            flags = FlagEffects(written=sem.flags_written, read=sem.flags_read)
            explanations = section.findall(".//explanations/explanation")
        except ExtractionError as e:
            failures[section_id] = str(e)
            continue
        section_category = _docvar(section, "instr-class")  # instr-class may sit at section or iclass level
        for ic in section.findall("classes/iclass"):
            category = _docvar(ic, "instr-class")
            if category is None:
                category = section_category
            if category is None:
                failures[f"{section_id} (iclass)"] = "no instr-class docvar"
                continue
            rd = ic.find("regdiagram")
            if rd is None:
                failures[f"{section_id} (iclass)"] = "iclass has no regdiagram"
                continue
            decode = _section_asl(ic, "Decode")  # Decode is per-iclass (addressing form: wback, scale, ...)
            boxes = {b.get("name"): _box_width(b) for b in rd.findall("box") if b.get("name")}
            for enc in ic.findall("encoding"):
                enc_name = enc.get("name")
                if not enc_name:
                    failures[f"{section_id} (encoding)"] = "encoding has no name"
                    continue
                try:
                    yield _build_instruction(section_id, category, explanations, asl, postdecode, sem,
                                             control_flow, flags, arch_constants, decode, boxes, rd, enc, enc_name)
                except ExtractionError as e:
                    failures[enc_name] = str(e)
