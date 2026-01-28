from typing import Optional, Tuple
from dataclasses import dataclass
from .bp import BP

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

        def __repr__(self):
            return f"<TwoBitEntry state={self.state} tag={self.tag}>"

    def __init__(self, number_of_bp_entries: int = 1024, tag_bits: int = 8, assoc: int = 8):
        if number_of_bp_entries & (number_of_bp_entries - 1) != 0:
            raise ValueError("number_of_bp_entries must be a power of 2")

        if assoc & (assoc - 1) != 0:
            raise ValueError("assoc must be a power of 2")

        self.index_bits: int = number_of_bp_entries.bit_length() - 1
        self.tag_mask: int = ((1 << tag_bits) - 1)
        self.index_mask: int = number_of_bp_entries - 1

        self.number_of_bp_entries: int = number_of_bp_entries
        self.assoc: int = assoc

        self.table: List[List[TwoBitBP.TwoBitEntry]] = [
            [TwoBitBP.TwoBitEntry() for _ in range(assoc)]
            for _ in range(number_of_bp_entries)
        ]

    def _index_and_tag(self, address: int) -> Tuple[int, int]:
        index   = (address >> 2) & self.index_mask
        tag     = (address >> (2 + self.index_bits)) & self.tag_mask
        return index, tag

    def _lookup(self, address: int) -> Optional[TwoBitEntry]:
        index, tag = self._index_and_tag(address)
        entries_in_index = self.table[index]

        for entry in entries_in_index:
            if tag == entry.tag:
                return entry

        return None

    def _access(self, address: int) -> TwoBitEntry:
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

        # Not found, replace LRU (last)
        entry = entries_in_index.pop()
        entry.tag = tag
        entry.state = 1  # weakly not taken
        entries_in_index.insert(0, entry)
        return entry

    def update(self, address: int, taken: bool) -> None:
        self._access(address).update(taken)

    def predict(self, address: int, update_state: bool = False) -> bool:
        get_entry = lambda a: self._lookup(a)
        if update_state:
            get_entry = lambda a: self._access(a)

        entry = get_entry(address)

        return entry.predict() if entry else False # Default Not-Taken

    def snapshot(self) -> Tuple[int, ...]:
        # TODO: Should I also include the tags?
        return tuple(
            tuple(set_entries)
            for set_entries in self.table
        )

    def __repr__(self):
        preview = ", ".join(
            str([entry.state for entry in set_entries])
            for set_entries in self.table[:min(5, self.number_of_bp_entries)]
        )
        return (
            f"<TwoBitBP entries={self.number_of_bp_entries} assoc={self.assoc} "
            f"index_bits={self.index_bits} preview={preview} ...>"
        )

    def __str__(self):
        return f"TwoBitBP({self.number_of_bp_entries} entries, {self.assoc}-way associative)"


    def print_table(self, max_sets: int = 8):
        print(f"TwoBitBP Table: {self.number_of_bp_entries} entries, {self.assoc}-way associative")
        sets_to_print = min(max_sets, self.number_of_bp_entries)

        for i in range(sets_to_print):
            entries = self.table[i]
            states_tags = [
                f"{entry.state}{f'({entry.tag})' if entry.tag is not None else ''}" 
                for entry in entries
            ]
            print(f"Set {i:03}: " + " | ".join(states_tags))

        if sets_to_print < self.number_of_bp_entries:
            print("...")

