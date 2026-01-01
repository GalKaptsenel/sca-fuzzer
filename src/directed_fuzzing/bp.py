from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List, Any, Tuple
from .microarch_event import MicroarchEvent

@dataclass
class BPMicroarchEvent(MicroarchEvent):
    pc: int
    taken: bool
    prediction: bool

class BP(ABC):
    @abstractmethod
    def update(self, address: int, taken: bool) -> None:
        pass

    @abstractmethod
    def predict(self, address: int) -> bool:
        pass

    @abstractmethod
    def snapshot(self) -> Any:
        """
        Return an immutable representation of the BP (all sets and entries)
        suitable for hashing or scoring.
        """
        pass


