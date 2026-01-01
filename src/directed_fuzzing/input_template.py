from __future__ import annotations
from typing import Optional, Dict, Callable, Union, List
import random
import copy
import numpy as np

from .input_generator import ValueGenerator, GEN_MAP

class InputConcrete:
    """Fully concrete, immutable instance of an InputTemplate."""
    def __init__(self, input_template: InputTemplate, rnd: Optional[random.Random] = None):
        self.input_template = input_template.clone()
        for cell in self.input_template.cells.values():
            if not cell.is_concrete():
                cell.set_concrete(rnd=rnd)

    def __getitem__(self, name: Union[str, int]) -> int:
        return self.input_template.get_cell(name).get_concrete()



class InputCell:
    """Abstract base class for any input (register, memory, etc.)"""
    def set_gen(self, gen: ValueGenerator) -> None:
        raise NotImplementedError()

    def set_params(self, params: Dict[str, any]) -> None:
        raise NotImplementedError()

    def get_concrete(self) -> int:
        raise NotImplementedError()

    def set_concrete(self, value: Optional[int] = None, rnd: Optional[random.Random] = None) -> None:
        raise NotImplementedError()

    def is_concrete(self) -> bool:
        raise NotImplementedError()

    def sample(self, n_samples: int = 1, rnd: Optional[random.Random] = None) -> List[int]:
        raise NotImplementedError()


class BaseCell(InputCell):
    """Represents a top-level input with a generator"""
    def __init__(self, name: Union[str, int], gen: ValueGenerator, params: Optional[Dict[str, any]] = None):
        self._name = name
        self._gen = gen
        self._params = params or {}
        self._value: Optional[int] = None

    def set_gen(self, gen: ValueGenerator) -> None:
        if self._value is not None:
            raise RuntimeError("Cannot modify concrete cell")
        self._gen = gen

    def set_params(self, params: Dict[str, any]) -> None:
        if self._value is not None:
            raise RuntimeError("Cannot modify concrete cell")

        assert isinstance(params, dict)

        merged = dict(self._params)
        merged.update(params)
        self._params = merged

    def get_concrete(self) -> int:
        if not self.is_concrete():
            raise RuntimeError(f"Cell {self._name} is not concrete")
        return self._value

    def set_concrete(self, value: Optional[int] = None, rnd: Optional[random.Random] = None) -> None:
        if self.is_concrete():
            raise RuntimeError(f"Cell {self._name} is already concrete")
 
        if value is None:
            value = self.sample(rnd=rnd)[0]

        value = np.uint64(value)
        assert isinstance(value, np.uint64), f"set_concrete with non-np.uint64 value ({type(value)=})"
        self._value = value


    def is_concrete(self) -> bool:
        return self._value is not None

    def sample(self, n_samples: int = 1, rnd: Optional[random.Random] = None) -> List[int]:
        if self.is_concrete():
            return [self.get_concrete()] * n_samples
        return [self._gen(self._params, rnd) for _ in range(n_samples)]


class SubCell(InputCell):
    """Represents a subset of another cell (e.g., w6 ⊂ x6)"""
    def __init__(self, name: Union[str, int], parent: BaseCell, mask: int):
        self._name = name
        self._parent = parent
        self._mask = mask

    def set_gen(self, gen: ValueGenerator) -> None:
        """
        WARNING:
        SubCells directly mutate their parent generator and params.
        No constraint tracking or refinement checks are performed.
        """
        self._parent.set_gen(gen)

    def set_params(self, params: Dict[str, any]) -> None:
        self._parent.set_params(params)

    def get_concrete(self) -> int:
        return self._parent.get_concrete() & self._mask

    def set_concrete(self, value: Optional[int] = None, rnd: Optional[random.Random] = None) -> None:
        if self._parent.is_concrete():
            raise RuntimeError("Parent cell already concrete")

        if value is None:
            value  = self.sample(rnd=rnd)[0]

        parent_val = self._parent.sample(1, rnd)[0]
        parent_val = np.uint64(parent_val)
        value      = np.uint64(value)
        mask       = np.uint64(self._mask)
        inv_mask = np.uint64(~mask)

        new_val = (parent_val & inv_mask) | (value & mask)
        self._parent.set_concrete(new_val, rnd)

    def is_concrete(self) -> bool:
        return self._parent.is_concrete()

    def sample(self, n_samples: int = 1, rnd: Optional[random.Random] = None) -> List[int]:
        return [val & self._mask for val in self._parent.sample(n_samples, rnd)]


class InputTemplate:
    """Static template describing all inputs and relationships"""
    def __init__(self):
        self.cells: Dict[str, InputCell] = {}

    def add_cell(self, name: Union[str, int], gen: ValueGenerator, params: Optional[Dict[str, any]] = None) -> BaseCell:
        if name in self.cells:
            raise ValueError(f"Cell {name} already exists")
        cell = BaseCell(name, gen, params)
        self.cells[name] = cell
        return cell

    def add_subcell(self, name: Union[str, int], parent_name: str, mask: int) -> SubCell:
        if name in self.cells:
            raise ValueError(f"Cell {name} already exists")
        if parent_name not in self.cells:
            raise ValueError(f"Parent cell {parent_name} not found")
        parent = self.cells[parent_name]
        cell = SubCell(name, parent, mask)
        self.cells[name] = cell
        return cell

    def instantiate(self, rnd: random.Random = None) -> InputConcrete:
        """Create a fresh runtime context for this template"""
        return InputConcrete(self, rnd)

    def get_cell(self, name: Union[str, int]) -> InputCell:
        cell = self.cells.get(name)
        if cell is None:
            raise KeyError(f"Cell {name} not found")
        return cell

    def clone(self) -> InputTemplate:
        return copy.deepcopy(self)


class InputTemplateBuilder:
    def __init__(self):
        self.template = InputTemplate()

    def add_cell_description(self, name: Union[str, int], gen: ValueGenerator, params: Optional[Dict[str, any]] = None, subcells: Optional[Dict[Union[str, int], int]] = None) -> InputTemplateBuilder:
        self.template.add_cell(name, gen, params)
        for subname, submask in (subcells or {}).items():
            self.template.add_subcell(subname, name, submask)

    def build(self) -> InputTemplate:
        """Finalize and return the template."""
        return self.template

