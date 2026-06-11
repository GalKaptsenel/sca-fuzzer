from __future__ import annotations
import glob
import os
from .download import download_xml
from .extract import iter_instructions
from .immediate import architectural_constants
from .serialize import write_json
from .models import ExtractionError


def build_json(out_json, xml_dir) -> dict:
    """Stages 2+3: parse every instruction XML in *xml_dir* and write *out_json*.
    Returns {filename: loud-fail message} for encodings not yet handled — they are skipped
    and reported, never silently wrong."""
    arch_constants = architectural_constants(os.path.join(str(xml_dir), "shared_pseudocode.xml"))
    instructions, failures = [], {}
    for f in sorted(glob.glob(os.path.join(str(xml_dir), "*.xml"))):
        file_failures: dict = {}
        for inst in iter_instructions(f, arch_constants, file_failures):
            instructions.append(inst)
        if file_failures:
            failures[os.path.basename(f)] = file_failures
    write_json(instructions, out_json)
    return failures


def run(out_json, release: str = "latest", cache_dir=None, force: bool = False) -> dict:
    """Full pipeline: download -> extract -> serialize. Returns the loud-fail report."""
    return build_json(out_json, download_xml(release, cache_dir, force))
