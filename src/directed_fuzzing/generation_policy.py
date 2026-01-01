import random
from abc import ABC, abstractmethod
from typing import Optional, List, Type

from .common import BranchType
from ..interfaces import Instruction


# ==================== Mixin Interfaces ====================

class DynamicInstructionLimitMixin(ABC):
    """
    Mixin interface for determining how many instructions are left to generate.

    Return value semantics:
    - None  : this mixin has no opinion
    - int>=0 : maximum remaining instructions (0 means stop immediately)
    """

    @abstractmethod
    def remaining_instructions(self, insts: List[Instruction]) -> Optional[int]:
        raise NotImplementedError


class BranchChooserMixin(ABC):
    """
    Mixin interface deciding what branch type should terminate the block.
    """

    @abstractmethod
    def choose_branch_type(self, insts: List[Instruction]) -> Optional[BranchType]:
        raise NotImplementedError


# ==================== Core Policy Interface ====================

class GenerationPolicy(ABC):
    """
    Interface for a concrete code block generation policy.
    """

    @abstractmethod
    def remaining_instructions(self, insts: List[Instruction]) -> int:
        raise NotImplementedError

    @abstractmethod
    def choose_branch_type(self, insts: List[Instruction]) -> BranchType:
        raise NotImplementedError


# ==================== Instruction Limit Mixins ====================

class MaxInstructionCountMixin(DynamicInstructionLimitMixin):
    """
    Stops when a block reaches a maximum number of instructions.
    """

    def __init__(self, max_instructions: int, **kwargs):
        super().__init__(**kwargs)
        if not isinstance(max_instructions, int) or max_instructions < 0:
            raise ValueError(
                f"max_instructions must be a non-negative integer (got {max_instructions})"
            )
        self._max_instructions = max_instructions

    def remaining_instructions(self, insts: List[Instruction]) -> Optional[int]:
        return max(self._max_instructions - len(insts), 0)


class EarlyStoppingInstructionCountMixin(DynamicInstructionLimitMixin):
    """
    Randomly stops a block with a given probability.
    """

    def __init__(
        self,
        early_stop_prob: float,
        rnd: Optional[random.Random] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        if not (0.0 <= early_stop_prob <= 1.0):
            raise ValueError(
                f"early_stop_prob must be in [0,1] (got {early_stop_prob})"
            )
        self._early_stop_prob = early_stop_prob
        self._rnd = rnd or random.Random()

    def remaining_instructions(self, insts: List[Instruction]) -> Optional[int]:
        if self._rnd.random() < self._early_stop_prob:
            return 0
        return None


# ==================== Branch Choice Mixins ====================

class RandomBranchTypeMixin(BranchChooserMixin):
    """
    Chooses a random branch type from a list of options.
    """

    def __init__(
        self,
        options: List[BranchType],
        rnd: Optional[random.Random] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        if not options or any(not isinstance(o, BranchType) for o in options):
            raise ValueError(
                f"options must be a non-empty list of BranchType (got {options})"
            )
        self._options = list(options)
        self._rnd = rnd or random.Random()

    def choose_branch_type(self, insts: List[Instruction]) -> BranchType:
        return self._rnd.choice(self._options)


# ==================== Combining Policy ====================

class CombiningGenerationPolicy(GenerationPolicy):
    """
    Combines all DynamicInstructionLimitMixin and BranchChooserMixin behaviors
    from its base classes.

    Instruction limit semantics:
    - All limits are combined using `min`
    - Any mixin may veto generation by returning 0
    """

    def __init_subclass__(cls):
        super().__init_subclass__()

        # Check that all concrete mixins appear before CombiningGenerationPolicy
        combining_index = cls.__mro__.index(CombiningGenerationPolicy)
        for mixin in (DynamicInstructionLimitMixin, BranchChooserMixin,
                      MaxInstructionCountMixin, EarlyStoppingInstructionCountMixin,
                      RandomBranchTypeMixin):
            if mixin in cls.__mro__ and cls.__mro__.index(mixin) > combining_index:
                raise TypeError(
                        f"{mixin.__name__} must appear before CombiningGenerationPolicy in bases"
                    )

    def _iter_concrete_mixins(
        self,
        mixin_type: Type[ABC],
        method_name: str,
    ):
        for base in type(self).__mro__:
            if (
                base is mixin_type
                or not issubclass(base, mixin_type)
                or method_name not in base.__dict__
            ):
                continue
            yield base

    def remaining_instructions(self, insts: List[Instruction]) -> int:
        remaining: Optional[int] = None

        for base in self._iter_concrete_mixins(
            DynamicInstructionLimitMixin, "remaining_instructions"
        ):
            limit = base.remaining_instructions(self, insts)
            if limit is None:
                continue
            remaining = limit if remaining is None else min(remaining, limit)

        if remaining is None:
            raise RuntimeError(
                "No DynamicInstructionLimitMixin provided a maximum instruction count"
            )

        return remaining

    def choose_branch_type(self, insts: List[Instruction]) -> BranchType:
        for base in self._iter_concrete_mixins(
            BranchChooserMixin, "choose_branch_type"
        ):
            branch = base.choose_branch_type(self, insts)
            if branch is not None:
                return branch

        raise RuntimeError(
            "No BranchChooserMixin provided a branch type"
        )


# ==================== Concrete Policies ====================

class RandomMaxInstRandomBranchGenerationPolicy(
    MaxInstructionCountMixin,
    EarlyStoppingInstructionCountMixin,
    RandomBranchTypeMixin,
    CombiningGenerationPolicy,
):
    """
    Max instructions + random early stop + random branch type.
    """

    def __init__(
        self,
        max_instructions: int,
        early_stop_prob: float,
        branch_options: List[BranchType],
        rnd: Optional[random.Random] = None,
    ):
        super().__init__(
            max_instructions=max_instructions,
            early_stop_prob=early_stop_prob,
            options=branch_options,
            rnd=rnd,
        )


class MaxInstRandomBranchGenerationPolicy(
    MaxInstructionCountMixin,
    RandomBranchTypeMixin,
    CombiningGenerationPolicy,
):
    """
    Max instructions + random branch type.
    """

    def __init__(
        self,
        max_instructions: int,
        branch_options: List[BranchType],
        rnd: Optional[random.Random] = None,
    ):
        super().__init__(
            max_instructions=max_instructions,
            options=branch_options,
            rnd=rnd,
        )

