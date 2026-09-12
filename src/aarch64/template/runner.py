"""Engine: turn a `Template` into a runnable `TestCase`, and load a template from a .py file.

`build_test_case` drives a caller-supplied generator (so the fuzzer's seed advances naturally);
`generate_test_case` makes its own. Both build a minimal entry function, run the template into it,
expand the holes, then reuse the generator's passes / printer / assembler."""
from __future__ import annotations

import importlib.util
import inspect

from ...interfaces import InstructionSetAbstract as InstructionSet
from ...interfaces import TestCase, Function, BasicBlock, LabelOperand
from ..aarch64_generator import Aarch64RandomGenerator, Aarch64IndirectCallPass
from .builder import Builder, Template
from .expand import expand_holes


def build_test_case(generator, template: Template, asm_file: str = "generated.asm",
                    assemble: bool = True) -> TestCase:
    # Seed handling mirrors ConfigurableGenerator.create_test_case: record the seed, then update_seed()
    # seeds `random` and advances the state -- so templates reproduce like the random path.
    test_case = TestCase(generator._state)
    generator.update_seed()
    generator.test_case = test_case
    generator.create_actors(test_case)
    actor = test_case.actors["main"]

    # entry function: the inline body ends by jumping to the test-case exit
    func = Function(".function_0", actor)
    func.append(BasicBlock(".bb_0.0"))
    func.exit.terminators = [
        generator.get_unconditional_jump_instruction().add_op(LabelOperand(test_case.exit.name))
    ]
    test_case.functions.append(func)

    entry = Builder(generator, test_case, func, func.get_first_bb())
    template.build(entry)
    entry._finalize()
    expand_holes(generator, test_case)

    # materialize branch terminators from each function's block graph, exactly as the random path does
    for f in test_case.functions:
        generator.add_terminators_in_function(f)

    for p in generator.passes:
        # The template materializes its own indirect calls (honoring per-call target counts), so the
        # random-path pass that would re-materialize every BLR is skipped.
        if isinstance(p, Aarch64IndirectCallPass):
            continue
        p.run_on_test_case(test_case)
    generator.add_required_symbols(test_case)

    test_case.asm_path = asm_file
    generator.printer.print(test_case, asm_file)
    if not assemble:
        return test_case

    bin_file = asm_file[:-4]
    obj_file = bin_file + ".o"
    generator.assemble(asm_file, obj_file, bin_file)
    test_case.bin_path = bin_file
    test_case.obj_path = obj_file
    test_case.symbol_table = []
    generator.get_elf_data(test_case, obj_file)
    return test_case


def generate_test_case(template: Template, instruction_set: InstructionSet, seed: int,
                       asm_file: str = "generated.asm", assemble: bool = True) -> TestCase:
    generator = Aarch64RandomGenerator(instruction_set, seed)
    generator.set_seed(seed)
    return build_test_case(generator, template, asm_file, assemble)


def load_template(path: str) -> Template:
    """Import a .py file and instantiate the single `Template` subclass it defines."""
    spec = importlib.util.spec_from_file_location("_revizor_user_template", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load template file: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    subclasses = [obj for _, obj in inspect.getmembers(module, inspect.isclass)
                  if issubclass(obj, Template) and obj is not Template
                  and obj.__module__ == module.__name__]
    if len(subclasses) != 1:
        raise ValueError(f"template file must define exactly one Template subclass, "
                         f"found {len(subclasses)} in {path}")
    return subclasses[0]()
