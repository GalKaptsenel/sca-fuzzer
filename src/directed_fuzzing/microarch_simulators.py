from abc import ABC, abstractmethod
from typing import List, Tuple

from ..interfaces import Instruction
from .microarch import MicroArchState

class SimulatorInterface(ABC):
    @abstractmethod
    def simulate(
        self,
        instructions: List[Instruction],
        input_template: InputTemplate,
        initial_state: MicroarchState
    ) -> Tuple[MicroarchState, InputTemplate]:
        """
        Simulate a sequence of instructions starting from initial_state,
        using InputTemplate to resolve operand values.
        """
        pass


class Simulator(SimulatorInterface):
    def __init__(self, arch_simulator):
        self.arch_sim = arch_simulator

    def step(self, instr: Instruction, mu_state: MicroarchState, ctx: InputTemplate) -> Tuple[MicroarchState, InputTemplate]:

        new_mu_state = copy.deepcopy(mu_state)

        new_ctx = self.arch_sim(instr, ctx)

        if instr.control_flow:

            target = new_ctx.target
            taken = new_ctx.taken

            new_mu_state.bp.update(target, taken)

        return new_mu_state, new_ctx

    def simulate(
        self,
        instructions: List[Instruction],
        input_template: InputTemplate,
        initial_state: MicroarchState
    ) -> Tuple[MicroarchState, InputTemplate]:
        state = initial_state
        ctx = input_template

        for instr in instructions:
            state, ctx = self.step(instr, state, ctx)

        return state, ctx


