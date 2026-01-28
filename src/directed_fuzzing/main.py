from .test_generator import generate_graph
from .block_graph import BlockGraph
from .mcts import MCTS
from .code_block import CodeAllocator
from .input_template import InputTemplate
from .aarch64_simulator import UnicornArchSimulator
from .microarch_simulators import MuSimulator
from .value_selector import DefaultValueSelectionStrategy
from .microarch_state import MicroarchState
from .two_bit_saturating_bp import TwoBitBP
from .scorer import NoveltyScorer
from .generation_policy import MaxInstRandomBranchGenerationPolicy
from .common import BranchType
from .instruction_generator import SimpleAArch64InstructionGenerator, SandboxMemoryTransform, TransformingInstructionGenerator
from .simulation import SimulationContext
from .arch_input_templates import build_aarch64_input_template

from ..isa_loader import InstructionSet
from ..aarch64.aarch64_target_desc import Aarch64TargetDesc
from ..config import CONF

def main():
    allocator = CodeAllocator(base_address=0x100000)
    graph = BlockGraph(allocator=allocator)

    fuzzed_registers = ["x0", "x1", "x2", "x3", "x4", "x5", "sp", "nzcv"]
    allowed_registers = [r for fuzzed_reg in fuzzed_registers
                          for r in Aarch64TargetDesc.reg_denormalized[
                                Aarch64TargetDesc.reg_normalized[fuzzed_reg]
                              ].values()
                    ]
    init_template = build_aarch64_input_template(Aarch64TargetDesc(), fuzzed_registers, 8192)

    value_selection_strategy = DefaultValueSelectionStrategy()
    arch_simulator = UnicornArchSimulator(value_strategy=value_selection_strategy)
    init_arch_snapshot = arch_simulator.take_snapshot()

    two_bit_bp = TwoBitBP()
    init_mu_state = MicroarchState(bp=two_bit_bp)

    init_context = SimulationContext(arch_snapshot=init_arch_snapshot, mu_state=init_mu_state)

    mu_simulator = MuSimulator(arch_simulator=arch_simulator)

    scorer = NoveltyScorer()

    generation_policy = MaxInstRandomBranchGenerationPolicy(max_instructions=8, branch_options=[BranchType.DIRECT_COND, BranchType.DIRECT_UNCOND])

    CONF.load('config.yml', '.')
    aarch64_instruction_set = InstructionSet("base.json", ['general'])
    aarch64_target_desc = Aarch64TargetDesc()
    instruction_gen = TransformingInstructionGenerator(
            SimpleAArch64InstructionGenerator(
                instruction_set=aarch64_instruction_set,
                target_description=aarch64_target_desc,
            ),
            transforms=[
                SandboxMemoryTransform(
                    sandbox_base_reg="x30",
                    sandbox_size=8192
                    )
                ]
            )

    mcts = MCTS(
            mu_simulator=mu_simulator,
            instr_generator=instruction_gen,
            scorer=scorer,
            initial_context_factory = lambda: init_context,
            initial_template_factory = lambda: init_template,
            generation_policy=generation_policy,
            verbose=True
        )

    block_node = generate_graph(graph=graph, block_gen=mcts, root_template=init_template, root_context=init_context)



if __name__ == '__main__':
    main()
