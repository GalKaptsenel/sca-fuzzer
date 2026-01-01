from abc import ABC, abstractmethod
from typing import List, Optional
from dataclasses import dataclass

from ..interfaces import Instruction
from .input_template import InputTemplate


@dataclass
class ArchSimStep:
    """
    Result of executing a single ISA instruction on the architectural simulator.

    Attributes:
        instruction_address: The address of the instruction being executed.
        updated_input_template: The InputTemplate after execution.
                                possibly adding constraints or updating symbolic/concrete values.
        taken:  If this instruction is control-flow, whether the branch was taken.
        target: If this instruction is control-flow, The target address of the branch,
    """
    instruction_address: int
    updated_input_template: InputTemplate
    taken: Optional[bool] = None
    target: Optional[int] = None


class ArchSnapshotInterface(ABC):
    @abstractmethod
    def restore(self):
        pass


class ArchSimulatorInterface(ABC):

    @abstractmethod
    def execute_instruction(
        self,
        instr: Instruction,
        input_template: InputTemplate
    ) -> ArchSimStep:
        """
        Execute a single instruction.

        Args:
            instr: Instruction to execute.
            input_template: Current InputTemplate with concrete/symbolic values.

        Returns:
            ArchSimStep containing updated InputTemplate and branch info.
        """
        pass

    def simulate(
        self,
        instructions: List[Instruction],
        input_template: InputTemplate
    ) -> InputTemplate:
        """
        Default simulation of a list of instructions using `execute_instruction`.

        Args:
            instructions: List of instructions to execute.
            input_template: Initial InputTemplate.

        Returns:
            Updated InputTemplate after all instructions executed.
        """
        template = input_template
        for instr in instructions:
            step = self.execute_instruction(instr, template)
            template = step.updated_input_template
        return template

    @abstractmethod
    def take_snapshot(self) -> ArchSnapshotInterface:
        pass

    @abstractmethod
    def restore_snapshot(self, snap: ArchSnapshotInterface):
        pass

