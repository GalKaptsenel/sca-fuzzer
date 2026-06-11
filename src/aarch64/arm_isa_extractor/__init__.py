"""Revizor-independent extractor of the ARM A64 ISA from ARM's XML (download -> parse -> JSON)."""
from .models import (
    Instruction, Operand, OperandKind, MemAccess, RegFile, FlagEffects,
    EncodingCtx, ExtractionError,
)
from .asl import AslSemantics, extract_asl_semantics
from .constraints import unpredictable_constraints
from .download import download_xml
from .extract import iter_instructions
from .serialize import write_json, instruction_dict
from .pipeline import build_json, run

__all__ = [
    "Instruction", "Operand", "OperandKind", "MemAccess", "RegFile", "FlagEffects",
    "EncodingCtx", "ExtractionError", "AslSemantics", "extract_asl_semantics",
    "unpredictable_constraints", "download_xml", "iter_instructions",
    "write_json", "instruction_dict", "build_json", "run",
]
