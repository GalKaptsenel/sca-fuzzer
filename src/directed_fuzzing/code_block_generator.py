from abc import ABC, abstractmethod
from typing import Tuple, List

from .input_template import InputTemplate
from .block_graph import BlockNode


class BlockGenerator(ABC):
    @abstractmethod
    def extend_block(
            self,
            block: BlockNode,
        ):
        """
        Extend the provided block with a newly generated sequence of instructions.
        Updates code_block's output fields
        """
        pass

