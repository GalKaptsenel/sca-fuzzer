from enum import Enum, auto


class BranchType(Enum):
    DIRECT_COND     = auto()
    DIRECT_UNCOND   = auto()
    INDIRECT        = auto()


class BranchOutcome(Enum):
    NOT_TAKEN = auto()
    TAKEN = auto()


