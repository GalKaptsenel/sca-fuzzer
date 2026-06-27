"""
File: Configuration factory

Copyright (C) Microsoft Corporation
SPDX-License-Identifier: MIT
"""

from typing import Tuple, Dict, List, Callable

from . import input_generator, analyser, postprocessor, interfaces
from .config import CONF, ConfigException
from .fuzzer import NoninterferenceFuzzer

# Arch-specific modules (.x86, .aarch64) and the Unicorn contract model (.model) are imported
# lazily inside the getters, keyed on the configured ISA.


def _get_from_config(options: Dict, key: str, conf_option_name: str, *args):
    GenCls = options.get(key, None)
    if GenCls:
        return GenCls(*args)

    raise ConfigException(
        f"ERROR: unknown value `{key}` of `{conf_option_name}` configuration option.\n"
        "  Available options are:\n  - " + "\n  - ".join(options.keys()))


def get_fuzzer(instruction_set, working_directory, testcase, inputs):
    if CONF.fuzzer == "architectural":
        if CONF.instruction_set == "x86-64":
            from .x86 import x86_fuzzer
            return x86_fuzzer.X86ArchitecturalFuzzer(instruction_set, working_directory, testcase,
                                                     inputs)
        raise ConfigException("ERROR: unknown value of `instruction_set` configuration option")
    elif CONF.fuzzer == "archdiff":
        if CONF.instruction_set == "x86-64":
            from .x86 import x86_fuzzer
            return x86_fuzzer.X86ArchDiffFuzzer(instruction_set, working_directory, testcase,
                                                inputs)
        raise ConfigException("ERROR: unknown value of `instruction_set` configuration option")
    elif CONF.fuzzer == "basic":
        if CONF.instruction_set == "x86-64":
            from .x86 import x86_fuzzer
            return x86_fuzzer.X86Fuzzer(instruction_set, working_directory, testcase, inputs)
        elif "aarch64" in CONF.instruction_set:
            from .aarch64 import aarch64_fuzzer
            return aarch64_fuzzer.Aarch64Fuzzer(instruction_set, working_directory, testcase, inputs)
        raise ConfigException("ERROR: unknown value of `instruction_set` configuration option")
    elif CONF.fuzzer == "non-interference":
        return NoninterferenceFuzzer(instruction_set, working_directory, testcase, inputs)

    raise ConfigException("ERROR: unknown value of `fuzzer` configuration option")


def get_program_generator(instruction_set: interfaces.InstructionSetAbstract,
                          seed: int) -> interfaces.Generator:
    key = CONF.instruction_set + "-" + CONF.generator
    if key == "aarch64-random":
        from .aarch64 import aarch64_generator
        return aarch64_generator.Aarch64RandomGenerator(instruction_set, seed)
    if key == "x86-64-random":
        from .x86 import x86_generator
        return x86_generator.X86RandomGenerator(instruction_set, seed)
    raise ConfigException(
        f"ERROR: unknown instruction_set/generator combination `{key}`")


def get_asm_parser(generator: interfaces.Generator) -> interfaces.AsmParser:
    if CONF.instruction_set == "x86-64":
        from .x86 import x86_asm_parser
        return x86_asm_parser.X86AsmParser(generator)
    if "aarch64" in CONF.instruction_set:
        from .aarch64 import aarch64_asm_parser
        return aarch64_asm_parser.Aarch64AsmParser(generator)
    raise ConfigException("ERROR: unknown value of `instruction_set` configuration option")


def get_input_generator(seed: int) -> interfaces.InputGenerator:
    if CONF.input_generator == "random":
        return input_generator.NumpyRandomInputGenerator(seed)
    if CONF.input_generator == "aarch64-nzcv":
        from .aarch64 import aarch64_input_generator
        return aarch64_input_generator.AArch64InputGenerator(seed)
    raise ConfigException(
        f"ERROR: unknown value `{CONF.input_generator}` of `input_generator` configuration option")


def get_model(bases: Tuple[int, int], enable_mismatch_check_mode: bool = False) -> interfaces.Model:
    # The contract model is Unicorn-based and x86-only (AArch64 uses the contract executor instead),
    # so Unicorn/x86 are imported here, lazily, rather than at module load.
    from . import model
    from .x86 import x86_model

    tracers = {
        "none": model.NoneTracer,
        "l1d": model.L1DTracer,
        "pc": model.PCTracer,
        "memory": model.MemoryTracer,
        "ct": model.CTTracer,
        "loads+stores+pc": model.CTTracer,
        "ct-nonspecstore": model.CTNonSpecStoreTracer,
        "ctr": model.CTRTracer,
        "arch": model.ArchTracer,
        "tct": model.TruncatedCTTracer,
        "tcto": model.TruncatedCTWithOverflowsTracer,
    }
    execution_clauses = {
        "seq": x86_model.X86UnicornSeq,
        "no_speculation": x86_model.X86UnicornSeq,
        "seq-assist": x86_model.X86SequentialAssist,
        "cond": x86_model.X86UnicornCond,
        "conditional_br_misprediction": x86_model.X86UnicornCond,
        "bpas": x86_model.X86UnicornBpas,
        "nullinj-fault": x86_model.X86UnicornNull,
        "nullinj-assist": x86_model.X86UnicornNullAssist,
        "delayed-exception-handling": x86_model.X86UnicornDEH,
        "div-zero": x86_model.X86UnicornDivZero,
        "div-overflow": x86_model.X86UnicornDivOverflow,
        "meltdown": x86_model.X86Meltdown,
        "fault-skip": x86_model.X86FaultSkip,
        "noncanonical": x86_model.X86NonCanonicalAddress,
        "vspec-ops-div": x86_model.x86UnicornVspecOpsDIV,
        "vspec-ops-memory-faults": x86_model.x86UnicornVspecOpsMemoryFaults,
        "vspec-ops-memory-assists": x86_model.x86UnicornVspecOpsMemoryAssists,
        "vspec-ops-gp": x86_model.x86UnicornVspecOpsGP,
        "vspec-all-div": x86_model.x86UnicornVspecAllDIV,
        "vspec-all-memory-faults": x86_model.X86UnicornVspecAllMemoryFaults,
        "vspec-all-memory-assists": x86_model.X86UnicornVspecAllMemoryAssists,
        "noninterference": x86_model.ActorNonInterferenceModel,
        "cond-bpas": x86_model.X86UnicornCondBpas,
        "cond-nullinj-fault": x86_model.X86NullInjCond,
    }

    # observational clause of the contract
    tracer = _get_from_config(tracers, CONF.contract_observation_clause,
                              "contract_observation_clause")

    # execution clause of the contract
    if "cond" in CONF.contract_execution_clause and "bpas" in CONF.contract_execution_clause:
        clause_name = "cond-bpas"
    elif "conditional_br_misprediction" in CONF.contract_execution_clause and \
            "nullinj-fault" in CONF.contract_execution_clause:
        clause_name = "cond-nullinj-fault"
    elif len(CONF.contract_execution_clause) == 1:
        clause_name = CONF.contract_execution_clause[0]
    else:
        raise ConfigException(
            "ERROR: unknown value of `contract_execution_clause` configuration option")

    return _get_from_config(execution_clauses, clause_name, "contract_execution_clause",
                            bases[0], bases[1], tracer, enable_mismatch_check_mode)


def get_executor(enable_mismatch_check_mode: bool = False,
                 generator: interfaces.Generator = None) -> interfaces.Executor:
    if CONF.executor in ("x86-64-intel", "x86-64-amd"):
        from .x86 import x86_executor
        cls = {"x86-64-intel": x86_executor.X86IntelExecutor,
               "x86-64-amd": x86_executor.X86AMDExecutor}[CONF.executor]
        return cls(enable_mismatch_check_mode)
    if CONF.executor == "aarch64":
        from .aarch64 import aarch64_executor
        # Regular fuzzing with PAC/MTE: each input runs its own genuine sealed TC (correct
        # signatures/tags for that input), so the box-resetting raw AUT*/tag-fault never happens.
        # Needs the generator (for the seal passes); falls back to the plain local executor otherwise.
        cats = CONF.instruction_categories or []
        if generator is not None and CONF.fuzzer != "non-interference" \
                and any(c.startswith(("PAC", "MTE")) for c in cats):
            return aarch64_executor.Aarch64RegularSealedExecutor(generator, enable_mismatch_check_mode)
        return aarch64_executor.Aarch64LocalExecutor(enable_mismatch_check_mode)
    raise ConfigException(
        f"ERROR: unknown value `{CONF.executor}` of `executor` configuration option")


def get_noninterference_executor(generator: interfaces.Generator,
                                  enable_mismatch_check_mode: bool = False) -> interfaces.Executor:
    """
    Create the non-interference executor (it auto-detects PAC/MTE from the enabled
    instruction categories) with the
    generator injected at construction.

    This is the only supported way to create a non-interference executor — the generator
    is a required constructor argument and cannot be set after the fact.

    Called by NoninterferenceFuzzer.initialize_modules() instead of get_executor().
    """
    if not CONF.instruction_set.startswith('aarch64'):
        raise ConfigException(
            f"ERROR: non-interference executor requires aarch64 instruction set; "
            f"got '{CONF.instruction_set}'")
    from .aarch64 import aarch64_executor
    # One non-interference executor; it auto-detects the active primitives (PAC and/or MTE) from the
    # enabled instruction categories and seals/compares whatever is present.
    return aarch64_executor.Aarch64NonInterferenceExecutor(generator, enable_mismatch_check_mode)


def get_analyser() -> interfaces.Analyser:
    analysers: Dict[str, type] = {
        'bitmaps': analyser.MergedBitmapAnalyser,
        'sets': analyser.SetAnalyser,
        'mwu': analyser.MWUAnalyser,
        'chi2': analyser.ChiSquaredAnalyser,
    }
    return _get_from_config(analysers, CONF.analyser, "analyser")


def get_minimizer(fuzzer: interfaces.Fuzzer,
                  instruction_set: interfaces.InstructionSetAbstract) -> interfaces.Minimizer:
    return postprocessor.MainMinimizer(fuzzer, instruction_set)


def get_downloader(arch: str, extensions: List[str], out_file: str) -> Callable:
    if arch == "x86-64":
        from .x86 import get_spec as x86_get_spec
        return x86_get_spec.Downloader(extensions, out_file)
    if arch == "aarch64":
        from .aarch64 import isa_downloader
        return isa_downloader.Downloader(extensions, out_file)
    raise ConfigException(f"ERROR: unknown value `{arch}` of `architecture` configuration option")
