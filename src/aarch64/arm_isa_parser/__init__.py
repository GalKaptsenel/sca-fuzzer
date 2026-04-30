"""
arm_isa_parser — ARM ISA XML downloader and parser for Revizor.

Typical usage (original Revizor interface)
------------------------------------------
    from arm_isa_parser import Downloader

    d = Downloader(["general"], "base.json")
    instructions = d.run()

Or call run() directly:

    from arm_isa_parser import run

    instructions = run("base.json", allowed_categories={"general"})

Or step by step (full control):

    from arm_isa_parser import ISADownloader, InstructionFileParser, save_json

    downloader = ISADownloader(release="A64-2025-09")
    instructions = []
    for xml_path in downloader.iter_xml_files():
        instructions.extend(InstructionFileParser(xml_path).parse())

    save_json(instructions, "base.json")
"""
from .models import InstructionSpec, OperandSpec
from .downloader import ISADownloader, KNOWN_RELEASES
from .parser import InstructionFileParser, generate_logical_immediates
from .pipeline import run, save_json, load_json, Downloader
from .tags import get_tags, tag_instructions

__all__ = [
    "InstructionSpec",
    "OperandSpec",
    "Downloader",
    "ISADownloader",
    "KNOWN_RELEASES",
    "InstructionFileParser",
    "generate_logical_immediates",
    "run",
    "save_json",
    "load_json",
]
