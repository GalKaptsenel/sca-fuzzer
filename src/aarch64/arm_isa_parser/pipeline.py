"""
Pipeline: download → parse → save.

This is the only file you need to import for a full end-to-end run.
All stages are independently usable for testing or custom workflows.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable, Optional

from .downloader import ISADownloader
from .models import InstructionSpec, OperandSpec
from .parser import InstructionFileParser
from .tags import tag_instructions
from .converter import to_rt_instructions

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Save to JSON
# ---------------------------------------------------------------------------

def save_json(instructions: list[InstructionSpec], path: Path | str) -> None:
    """Write *instructions* as a JSON array to *path*."""
    path = Path(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump([inst.to_dict() for inst in instructions], f, indent=2)
    logger.info("Wrote %d instructions to %s", len(instructions), path)


def load_json(path: Path | str) -> list[InstructionSpec]:
    """Reload a previously saved JSON file into InstructionSpec objects."""
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        raw = json.load(f)

    instructions = []
    for item in raw:
        inst = InstructionSpec(
            name=item["name"],
            category=item["category"],
            control_flow=item["control_flow"],
            datatype=item.get("datatype", ""),
            template=item["template"],
            operands=[_operand_from_dict(o) for o in item["operands"]],
            implicit_operands=[_operand_from_dict(o) for o in item["implicit_operands"]],
            tags=item.get("tags", []),
        )
        instructions.append(inst)
    return to_rt_instructions(instructions)


def _operand_from_dict(d: dict) -> OperandSpec:
    return OperandSpec(
        name=d["name"],
        type_=d["type_"],
        values=d["values"],
        src=d["src"],
        dest=d["dest"],
        width=d["width"],
        signed=d["signed"],
    )


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

class Downloader:
    """
    High-level wrapper matching the original Revizor interface.

        d = Downloader(["general"], "base.json")
        instructions = d.run()           # returns list[InstructionSpec]
        rt_instructions = d.run_rt()     # returns list[RT_InstructionSpec]
    """

    def __init__(self, extensions: list[str], out_file: str) -> None:
        self.extensions = extensions  # instr-class categories, e.g. ["general"]
        self.out_file = out_file

    def run(self) -> list[InstructionSpec]:
        return run(
            output_path=self.out_file,
            allowed_categories=set(self.extensions) if self.extensions else None,
        )

    def run_rt(self):
        """Return instructions as Revizor-native RT_InstructionSpec objects."""
        from .converter import to_rt_instructions
        return to_rt_instructions(self.run())


def run(
    output_path: Path | str = "base.json",
    *,
    release: str = "latest",
    cache_dir: Optional[Path] = None,
    force_download: bool = False,
    allowed_categories: Optional[set[str]] = None,
    verbose: bool = False,
) -> list[InstructionSpec]:
    """
    Full pipeline: download → parse → save → return.

    Parameters
    ----------
    output_path:
        Where to write the final JSON.
    release:
        ARM ISA release identifier (see downloader.KNOWN_RELEASES).
    cache_dir:
        Override the default cache location.
    force_download:
        Re-download even if cached files exist.
    allowed_categories:
        Filter to specific instr-class values, e.g. ``{"general"}``.
        Pass ``None`` to include all categories.
        each unique instruction shape appears once with a full value union.
    verbose:
        If True, set logging to INFO for this module.

    Returns
    -------
    list[InstructionSpec]
        The final list of parsed instructions.
    """
    if verbose:
        logging.basicConfig(level=logging.INFO)

    downloader = ISADownloader(release=release, cache_dir=cache_dir, force_download=force_download)
    xml_files = downloader.xml_files()
    logger.info("Found %d XML files", len(xml_files))

    all_instructions: list[InstructionSpec] = []
    for i, xml_path in enumerate(xml_files, 1):
        logger.info("[%d/%d] Parsing %s", i, len(xml_files), xml_path.name)
        parser = InstructionFileParser(xml_path, allowed_categories=allowed_categories)
        all_instructions.extend(parser.parse())

    logger.info("Parsed %d raw encodings", len(all_instructions))

    tag_instructions(all_instructions)
    logger.info("Tagging complete")

    save_json(all_instructions, output_path)
    return all_instructions

