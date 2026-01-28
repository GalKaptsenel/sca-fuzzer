from typing import Optional, Dict, List
from .input_template import InputTemplateBuilder, InputTemplate
from .input_generator import GEN_MAP, ValueGenerator
from ..interfaces import TargetDesc

class ArchInputTemplateBuilder:
    def __init__(self, target_description: TargetDesc):
        self._target_desc = target_description
        self._builder = InputTemplateBuilder()

    def add_reg(self, name: str, gen: ValueGenerator, params: Optional[Dict[str, any]] = None):
        if name not in self._target_desc.register_sizes:
            raise RuntimeError(f"Unexpected register name: {name}")

        reg_variants = self._target_desc.reg_denormalized[self._target_desc.reg_normalized[name]]
        subcells: Dict[str, int] = {}
        for size, subname in reg_variants.items():
            subcells[subname] = (1 << size) - 1

        master_cell_name = f"{name}_master"
        self._builder.add_cell_description(master_cell_name, gen, params, subcells)

    def add_cell(self, name: str, gen: ValueGenerator, params: Optional[Dict[str, any]] = None, subcells: Optional[Dict[str, int]] = None):
        self._builder.add_cell_description(name, gen, params, subcells)

    def build(self) -> InputTemplate:
        return self._builder.build()


def build_aarch64_input_template(target_description: TargetDesc, regs: List[str], memory_size: int) -> InputTemplate:
    builder = ArchInputTemplateBuilder(target_description)

    for reg in regs:
        builder.add_reg(reg, GEN_MAP['random_64'])

    for offset in range(0, memory_size, 8):
        #name = f"mem_0x{offset:X}"
        name = offset
        builder.add_cell(name, GEN_MAP['random_64_low_entropy'])

    return builder.build()

