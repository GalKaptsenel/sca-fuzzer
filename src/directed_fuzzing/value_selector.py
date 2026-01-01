from typing import List, Protocol, Union
from .input_template import InputTemplate
from ..interfaces import Input


class ValueSelectionStrategy(Protocol):
    def __call__(
            self,
            name: Union[str, int],
            template: InputTemplate,
            n_candidates: int = 1,
        ) -> int:
        """
        Choose a single value for a given input name from the template.
        """
        ...


class DefaultValueSelectionStrategy:
    def __call__(
        self,
        name: Union[str, int],
        template: InputTemplate,
        n_candidates: int = 1,
    ) -> int:
        if n_candidates < 1:
            raise ValueError(
                f"Number of candidates to sample from {name} must be positive, got {n_candidates}"
            )

        candidates: List[int] = template.get_cell(name).sample(n_candidates)

        if not candidates:
            raise ValueError(f"No candidates returned for {name} from template")

        # For now, just take the first one generated
        return candidates[0]


class InputForwarder:
    def __init__(self, forwarded: Input):
        self._forwarded = forwarded

    def __call__(
        self,
        name: Union[str, int],
        template: InputTemplate,
        n_candidates: int = 1,
    ) -> int:
        if n_candidates < 1:
            raise ValueError(
                f"Number of candidates to sample from {name} must be positive, got {n_candidates}"
            )

        if isinstance(name, str):
            offset = int(name[-1])
            return self._forwarded[0]['gpr'][offset]
        elif isinstance(name, int):
            offset = name
            if 0 <= name < 4096:
                field_name = 'main'
                offset = name
            elif 4096 <= name < 2 * 4096:
                field_name = 'faulty'
                offset = (name - 4096) // 8

            offset //= self._forwarded[0][field_name].itemsize
            return self._forwarded[0][field_name][offset]
        else:
            raise RuntimeError(f"Unsupported name type: {type(name)}")

