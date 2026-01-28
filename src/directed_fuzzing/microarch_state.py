from __future__ import annotations
from typing import Any
import copy
from .bp import BP

class MicroarchState:
    """
    microarchitectural state representation.
    """

    def __init__(self, bp: BP):
        self.bp = bp

    def snapshot(self) -> Any:
        """Return an immutable representation of the state suitable for hashing."""
        return self.bp.snapshot()

    def clone(self) -> MicroarchState:
        return copy.deepcopy(self)

