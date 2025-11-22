import random
from abc import ABC, abstractmethod
from typing import Optional, List

from .common import BranchType
from ..interfaces import Instruction


class DynamicInstructionLimitMixin(ABC):
    """Mixin interface for determining how many instruction left to generate."""
    @abstractmethod
    def remaining_instructions(self, insts: List[Instruction]) -> Optional[int]:
        pass


class BranchChooserMixin(ABC):
    """Mixin interface deciding what branch type should terminate the block."""
    @abstractmethod
    def choose_branch_type(self, insts: List[Instruction]) -> Optional[BranchType]:
        pass


class GenerationPolicy(ABC):
    """Interface for a code block generation policy."""
    @abstractmethod
    def remaining_instructions(self, insts: List[Instruction]) -> int:
        pass

    @abstractmethod
    def choose_branch_type(self, insts: List[Instruction]) -> BranchType:
        pass


class MaxInstructionCountMixin(DynamicInstructionLimitMixin):
    """Stops when a block reaches a maximum number of instructions."""
    def __init__(self, max_instructions: int, **kwargs):
        super().__init__(**kwargs)
        if not isinstance(max_instructions, int) or max_instructions < 0:
            raise ValueError(f"max_instructions must a non-negative integer (got {max_instructions})")
        self._max_instructions = max_instructions

    def remaining_instructions(self, insts: List[Instruction]) -> Optional[int]:
        return self._max_instructions - len(insts)


class EarlyStoppingInstructionCountMixin(DynamicInstructionLimitMixin):
    """Randomly stops a block with a given probability."""
    def __init__(self, early_stop_prob: float, rnd: Optional[random.Random] = None, **kwargs):
        super().__init__(**kwargs)
        if not (0 <= early_stop_prob <= 1):
            raise ValueError(f"early_stop_prob must be in [0,1] (got {early_stop_prob})")
        self._early_stop_prob = early_stop_prob
        self._rnd = rnd or random.Random()

    def remaining_instructions(self, insts: List[Instruction]) -> Optional[int]:
        if self._rnd.random() < self._early_stop_prob:
            return 0
        return None


class RandomBranchTypeMixin(BranchChooserMixin):
    """Chooses a random branch type from a list of options."""
    def __init__(self, options: List[BranchType], **kwargs):
        super().__init__(**kwargs)
        if not options or any(not isinstance(o, BranchType) for o in options):
            raise ValueError(f"options must be a non-empty list of BranchType (got {options})")
        self._options = list(options)

    def choose_branch_type(self, insts: List[Instruction]) -> BranchType:
        return random.choice(self._options)


# -------------------- Combining Mixin --------------------

class CombiningGenerationPolicy(GenerationPolicy):
    """
    Automatically combines all DynamicInstructionLimitMixin and BranchChooserMixin behaviors
    from its base classes.
    """
    def remaining_instructions(self, insts: List[Instruction]) -> int:
        remaining = None
        for base in type(self).__mro__:
            if issubclass(base, DynamicInstructionLimitMixin) and base is not DynamicInstructionLimitMixin:
                limit: Optional[int] = base.remaining_instructions(self, insts)
                if limit is None:
                    continue
                if remaining is None:
                    remaining = limit
                else:
                    remaining = min(limit, remaining)

        if remaining is None:
            raise RuntimeError("None of the DynamicInstructionLimitMixin provided a max instruction number!")

        assert isinstance(remaining, int)
        return remaining


    def choose_branch_type(self, insts: List[Instruction]) -> BranchType:
        for base in type(self).__mro__:
            if issubclass(base, BranchChooserMixin) and base is not BranchChooserMixin:
                branch = base.choose_branch_type(self, insts)
                if branch is not None:
                    return branch
        raise RuntimeError("No BranchChooserMixin provided a branch type")


# -------------------- Concrete Policies --------------------

class RandomMaxInstRandomBranchGenerationPolicy(
        MaxInstructionCountMixin,
        EarlyStoppingInstructionCountMixin,
        RandomBranchTypeMixin,
        CombiningGenerationPolicy
    ):
    """Max instructions + random early stop + random branch type."""
    def __init__(self, max_instructions: int, early_stop_prob: float, branch_options: List[BranchType], rnd: Optional[random.Random] = None):
        super().__init__(
            max_instructions=max_instructions,
            early_stop_prob=early_stop_prob,
            options=branch_options,
            rnd=rnd
        )


class MaxInstRandomBranchGenerationPolicy(
        MaxInstructionCountMixin,
        RandomBranchTypeMixin,
        CombiningGenerationPolicy
    ):
    """Max instructions + random branch type."""
    def __init__(self, max_instructions: int, branch_options: List[BranchType], rnd: Optional[random.Random] = None):
        super().__init__(
            max_instructions=max_instructions,
            options=branch_options,
            rnd=rnd
        )

