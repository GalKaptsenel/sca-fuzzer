from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List, Any

class BP(ABC):
    @abstractmethod
    def update(self, address: int, taken: bool) -> None:
        pass

    @abstractmethod
    def predict(self, address: int) -> bool:
        pass

    def snapshot(self) -> Any
        """
        Return an immutable representation of the BP (all sets and entries)
        suitable for hashing or scoring.
        """
        pass


class TwoBitBP(BP):
    @dataclass
    class TwoBitEntry:
        """
        2-bit saturating counter branch predictor entry with optional tag.
        States: 0=strongly not taken, 1=weakly not taken,
                2=weakly taken, 3=strongly taken
        """

        state: int = 1
        tag: Optional[int] = None
    
        def predict(self) -> bool:
            return self.state >= 2
    
        def update(self, taken: bool):
            if taken:
                self.state = min(3, self.state + 1)
            else:
                self.state = max(0, self.state - 1)

    def __init__(self, number_of_bp_entries: int = 1024, tag_bits: int = 8, assoc: int = 8):
        if number_of_bp_entries & (number_of_bp_entries - 1) != 0:
            raise ValueError("number_of_bp_entries must be a power of 2")

        if assoc & (assoc - 1) != 0:
            raise ValueError("assoc must be a power of 2")

        self.tag_mask: int = (1 << tag_bits) - 1
        self.index_bits = number_of_bp_entries.bit_length() - 1
        self.index_mask: int = number_of_bp_entries - 1

        self.number_of_bp_entries: int = number_of_bp_entries
        self.assoc = assoc

        self.table: List[List[TwoBitBP.TwoBitEntry]] = [[TwoBitBP.TwoBitEntry() for _ in range(assoc)] for _ in range(number_of_bp_entries)]

    def _index_and_tag(self, address: int) -> Tuple[int, int]:
        index = address & self.index_mask
        tag = (address >> self.index_bits) & self.tag_mask
        return index, tag

    def _access(self, address: int) -> TwoBitBP.TwoBitEntry:
        """
        Find the respective entry in the BP. 
        If couldn't find, replace the LRU entry in the corresponding set.
        """
        index, tag = self._index_and_tag(address)
        entries_in_index = self.table[index]
        for entry_index, entry in enumerate(entries_in_index):
            if tag == entry.tag:
                entries_in_index.insert(0, entries_in_index.pop(entry_index))
                return entries_in_index[0]

        # Not Found, Remove LRU
        entry = entries_in_index.pop()
        entry.tag = tag
        entry.state = 1  # weakly not taken
        entries_in_index.insert(0, entry)
        return entry

    def update(self, address: int, taken: bool) -> None:
        self._access(address).update(taken)

    def predict(self, address: int) -> bool:
        return self._access(address).predict()

    def snapshot(self) -> Tuple[int, ...]:
        return tuple(
                entry.state 
                for set_entries in self.table
                for entry in set_entries
        )

