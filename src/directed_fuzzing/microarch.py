from collections import NamedTuple
from .bp import BP

class MicroarchState(NamedTuple):
    """
    Opaque microarchitectural state representation.
    """
    bp: BP

    def snapshot(self) -> Any:
        """Return an immutable representation of the state suitable for hashing."""
        return self.bp.snapshot()

