from __future__ import annotations
from typing import Optional, Tuple, Callable, List
from collections import OrderedDict
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List, Any, Tuple
from functools import reduce

class BP(ABC):
    @abstractmethod
    def update(self, address: int, taken: bool, target: Optional[int] = None) -> None:
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


class LRUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.d = OrderedDict()

    def get(self, key, quiet: bool = False):
        if key not in self.d:
            return None
        if not quiet:
            self.d.move_to_end(key)
        return self.d[key]

    def put(self, key, value):
        if key in self.d:
            self.d.move_to_end(key)
        self.d[key] = value
        if len(self.d) > self.capacity:
            self.d.popitem(last=False)
    def __repr__(self):
        items = ", ".join(f"{k:#X}: {v}" for k, v in self.d.items())
        return f"LRUCache(capacity={self.capacity}, [{items}])"


# It can not implement BP as it does not garantee a prediction
class SaturatingCounterBP:
    class NBitCounterEntry:

        def __init__(self, n: int, initial_state: Optional[int] = None):
            self._n: int = n
            self._state = initial_state if initial_state is not None else (1 << (n - 1))
            self._min = 0
            self._max = (1 << self._n) - 1
            self._threshold = (1 << (self._n - 1))

        def predict(self) -> bool:
            return self._state >= self._threshold

        def update(self, taken: bool):
            if taken:
                self._state = min(self._max, self._state + 1)
            else:
                self._state = max(self._min, self._state - 1)

        def __repr__(self):
            return f"<NBitCounterEntry n={self._n} state={self._state}>"

        def snapshot(self) -> int:
            return self._state

    class SetEntry:
        def __init__(self, associativity: int, entry_factory: Callable[[], SaturatingCounterBP.NBitCounterEntry], tag_fn: Optional[Callable[[int], int]] = None):

            if associativity & (associativity - 1) != 0:
                raise ValueError("assoc must be a power of 2")

            if associativity > 1 and tag_fn is None:
                raise ValueError("associativity is bigger then 1. tag_fn must be supplied!")

            self._entry_factory = entry_factory
            self._tag_fn: Callable[int, int] = tag_fn or (lambda x: 0)
            self._lru_cache: LRUCache = LRUCache(associativity)

        def lookup(self, address: int, update_cache: bool = True) -> Optional[SaturatingCounterBP.NBitCounterEntry]:
            return self._lru_cache.get(self._tag_fn(address), not update_cache)

        def insert(self, address: int) -> SaturatingCounterBP.NBitCounterEntry:
            entry = self._entry_factory()
            self._lru_cache.put(self._tag_fn(address), entry)
            return entry

        def __repr__(self):
            return f"<NSetEntry lru_cache={self._lru_cache}>"

        def snapshot(self) -> Tuple[Tuple[int, int], ...]:
            return tuple(
                    (k, v.snapshot()) for k, v in self._lru_cache.d.items()
            )

    def __init__(self, counter_bit_width: int,  num_sets: int, assoc: int = 1, index_fn: Optional[Callable[[int], int]] = None, tag_fn: Optional[Callable[[int], int]] = None):
        if num_sets & (num_sets- 1) != 0:
            raise ValueError("num_sets must be a power of 2")

        if num_sets > 1 and index_fn is None:
            raise ValueError("number of sets is bigger then 1. index_fn must be supplied!")

        self._index_fn: Callable[int, int] = index_fn or (lambda x: 0)

        self._table: List[SaturatingCounterBP.SetEntry] = [SaturatingCounterBP.SetEntry(assoc, lambda: SaturatingCounterBP.NBitCounterEntry(counter_bit_width), tag_fn) for _ in range(num_sets)]

    def _lookup(self, address: int, update_set: bool, allocate_on_miss: bool) -> Optional[SaturatingCounterBP.NBitCounterEntry]:
        idx = self._index_fn(address)
        assert isinstance(idx, int) and 0 <= idx < len(self._table)
        set_entry = self._table[idx]
        counter = set_entry.lookup(address, update_set)
        if counter is None:
            if not allocate_on_miss:
                return None
            counter = set_entry.insert(address)
        return counter

    def update(self, address: int, taken: bool, touch_lru: bool = True) -> None:
        self._lookup(address, touch_lru, True).update(taken)

    def predict(self, address: int, touch_lru: bool = False, allocate_on_miss: bool = False) -> Optional[bool]:
        entry = self._lookup(address, touch_lru, allocate_on_miss)
        if entry is None:
            return None
        return entry.predict()

    def snapshot(self) -> Tuple[Tuple[Tuple[int, int], ...], ...]:
        return tuple(
            set_entry.snapshot()
            for set_entry in self._table
        )

    def __repr__(self):
        preview_len = 5
        preview = ", ".join(
            str(set_entry)
            for set_entry in self._table[:min(preview_len, len(self._table))]
        )

        if len(self._table) > preview_len:
            preview += ", ..."

        return f"<SaturatingCounterBP #indexes={len(self._table)} preview={preview}>"

    def print_table(self, max_sets: Optional[int] = None):
        num_sets = len(self._table)
        max_sets = max_sets or num_sets
        sets_to_print = min(max_sets, num_sets)

        for i in range(sets_to_print):
            set_entry = self._table[i]
            print(f"Set {i:03}: {set_entry}")

        if sets_to_print < num_sets:
            print("...")


class SaturatingCounterBPCommon(BP):
    def __init__(self, counter_bit_width: int,  num_sets: int, assoc: int = 1):
        index_bits = (num_sets - 1).bit_length()
        pc_shift = 2
        index_fn = lambda pc: (pc >> pc_shift) & (num_sets - 1)
        tag_fn = lambda pc: pc >> (pc_shift + index_bits)
        self._bp = SaturatingCounterBP(counter_bit_width, num_sets, assoc, index_fn, tag_fn)

    def update(self, address: int, taken: bool, touch_lru: bool = True) -> None:
        self._bp.update(address, taken, touch_lru)

    def predict(self, address: int, touch_lru: bool = False) -> bool:
        return self._bp.predict(address, touch_lru, True)

    def snapshot(self) -> Tuple[Tuple[Tuple[int, int], ...], ...]:
        return self._bp.snapshot()

    def __repr__(self):
        return repr(self._bp)

    def print_table(self, max_sets: Optional[int] = None):
        self._bp.print_table(max_sets)


class ShiftRegister:
    def __init__(self, n_bits: int):
        self._n_bits = n_bits
        self._mask = (1 << n_bits) - 1
        self._value = 0

    def update(self, bit: int):
        self._value = ((self._value << 1) | (bit & 1)) & self._mask

    def read(self) -> int:
        return self._value

    def __repr__(self):
        return f"{self._value:0{self._n_bits}b}"


class GHR(ShiftRegister):
    def update(self, taken: bool):
        super().update(1 if taken else 0)


class PHR:
    def __init__(self, n_bits: int, update_fn: Callable[[int, int, int], int]):
        self._n_bits = n_bits
        self._mask = (1 << n_bits) - 1
        self._value = 0
        self._update_fn = update_fn

    def update(self, pc: int, target: int):
        self._value = self._update_fn(self._value, pc, target) & self._mask

    def read(self) -> int:
        return self._value

    def __repr__(self):
        return f"{self._value:0{self._n_bits}b}"


class TAGEBase(BP):
    def __init__(self, counter_bit_width: int,  num_sets: int, assoc: int = 1):
        index_bits = (num_sets - 1).bit_length()
        pc_shift = 2
        index_fn = lambda pc: (pc >> pc_shift) & (num_sets - 1)
        tag_fn = lambda pc: pc >> (pc_shift + index_bits)
        self._bp = SaturatingCounterBP(counter_bit_width, num_sets, assoc, index_fn, tag_fn)

    def update(self, address: int, taken: bool, touch_lru: bool = True) -> None:
        self._bp.update(address, taken, touch_lru)

    def predict(self, address: int, touch_lru: bool = False) -> bool:
        return self._bp.predict(address, touch_lru, True)

    def snapshot(self) -> Tuple[Tuple[Tuple[int, int], ...], ...]:
        return self._bp.snapshot()

    def __repr__(self):
        return repr(self._bp)

    def print_table(self, max_sets: Optional[int] = None):
        self._bp.print_table(max_sets)


# It can not implement BP as it does not garantee a prediction
class TAGEPHT:
    def __init__(self, counter_bit_width: int,  num_sets: int, assoc: int, index_fn: Callable[[int], int], tag_fn: Callable[[int], int]):
        self._bp = SaturatingCounterBP(counter_bit_width, num_sets, assoc, index_fn, tag_fn)

    def update(self, address: int, taken: bool, touch_lru: bool = True) -> None:
        self._bp.update(address, taken, touch_lru)

    def predict(self, address: int, touch_lru: bool = False) -> Optional[bool]:
        return self._bp.predict(address, touch_lru, False)

    def snapshot(self) -> Tuple[Tuple[Tuple[int, int], ...], ...]:
        return self._bp.snapshot()

    def __repr__(self):
        return repr(self._bp)

    def print_table(self, max_sets: Optional[int] = None):
        self._bp.print_table(max_sets)


class Aarch64NeoverseN3BPU(BP):
    def __init__(self):
        def phr_footprint(pc: int, target: int) -> int:
            def get_bit(x: int, k: int) -> int:
                return (x >> k) & 1
            b0 = get_bit(pc, 2) ^ get_bit(pc, 6) ^ get_bit(target, 3) ^ get_bit(target, 7)
            b1 = get_bit(pc, 3) ^ get_bit(pc, 7) ^ get_bit(target, 4) ^ get_bit(target, 8)
            b2 = get_bit(pc, 4) ^ get_bit(pc, 8) ^ get_bit(target, 5) ^ get_bit(target, 9)
            b3 = get_bit(pc, 5) ^ get_bit(pc, 9) ^ get_bit(target, 6) ^ get_bit(target, 10)
            return (b0 << 0) ^ (b1 << 1) ^ (b2 << 2) ^ (b3 << 3)


        self._histlen: List[int] = [0, 172, 300]

        num_sets_base = (1 << 12)
        num_sets_phts = (1 << 11)
        counter_bit_width: int = 3
        self._phr: PHR = PHR(self._histlen[-1], lambda phr, pc, target: (phr << 4) ^ phr_footprint(pc, target))

        def generate_index_fn(phr_len: int) -> int:
            def index_fn(address: int):
                return (address ^ (self._phr.read() & ((1 << phr_len) - 1))) & (num_sets_phts - 1)
            return index_fn

        def generate_tag_fn(phr_len: int) -> int:
            def tag_fn(address: int):
#                value = self._phr.read()
#                result = 0
#                phr_fold_length = 11
#                phr_mask = (1 << fold_length) - 1
#                for i in range(phr_len, fold_length - 1, -1 * fold_length):
#                    result ^= (value >> (i - fold_length)) & mask
#                phr_b0 = (result >> (phr_fold_length - 1)) & 1
#                phr_b1 = (result >> (phr_fold_length - 2)) & 1
#
#                address_bit_mask = (1 << 8) - 1
#
#                return (address & address_bit_masj) ^ ()

                return address ^ (self._phr.read() & ((1 << phr_len) - 1))
            return tag_fn

        self._phts = [
                TAGEBase(counter_bit_width, num_sets_base),
                TAGEPHT(counter_bit_width, num_sets_phts, 4, generate_index_fn(self._histlen[1]), generate_tag_fn(self._histlen[1])),
                TAGEPHT(counter_bit_width, num_sets_phts, 2, generate_index_fn(self._histlen[2]), generate_tag_fn(self._histlen[2]))
        ]

    def update(self, address: int, taken: bool, target: int, touch_lru: bool = True) -> None:
        flag = 0
        for idx, pht in reversed(list(enumerate(self._phts))):
            prediction = pht.predict(address, touch_lru)
            if prediction is not None:
                flag += 1
                pht.update(address, taken, touch_lru)
                if prediction != taken and idx + 1 < len(self._phts):
                    self._phts[idx + 1].update(address, taken, touch_lru)
                break

        assert flag == 1, "Should update exactly once"

        if taken:
            self._phr.update(address, target)

    def predict(self, address: int, touch_lru: bool = False) -> bool:
        for pht in reversed(self._phts):
            prediction = pht.predict(address, touch_lru)
            if prediction is not None:
                return prediction
        assert False, "Should not reach here"

    def snapshot(self) -> Tuple[Tuple[Tuple[int, int], ...], ...]:
        return reduce(lambda acc, pht: acc + pht.snapshot(), self._phts, ())

    def __repr__(self):
        return reduce(lambda acc, pht: acc + "\n" + repr(pht), self._phts, "")

    def print_table(self, max_sets: Optional[int] = None):
        for idx, pht in enumerate(self._phts):
            print(f"Table at index {idx} ({'Base Predictor' if idx == 0 else f'Tagged Table {idx} out of {len(self._phts) - 1} with history of length {self._histlen[idx]}'})")
            pht.print_table(max_sets)

def main():
    bpu = Aarch64NeoverseN3BPU()
    bpu.update(0x1234, True, 0x1234)
    bpu.update(0x1234, True, 0x1234)
    import pdb; pdb.set_trace()
    bpu.update(0x1234, False, 0x1234)
    for _ in range(300):
        bpu.update(0x4444, True, 0x8888)

    bpu.update(0x1234, True, 0x1234)
    bpu.update(0x1234, True, 0x1234)
    import pdb; pdb.set_trace()
    bpu.update(0x1234, True, 0x1234)
    import pdb; pdb.set_trace()

if __name__ == "__main__":
    main()
