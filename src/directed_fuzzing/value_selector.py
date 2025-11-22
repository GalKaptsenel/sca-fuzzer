from abc import ABC, abstractmethod
from typing import List
from .input_template import InputTemplate


class ValueSelectionStrategy(ABC):
    @abstractmethod
    def choose_value(
            self,
            name: str,
            template: InputTemplate,
            n_samples: int = 1,
        ) -> int:
        """
        Choose a single value for a given input name from the template.
        """
        pass


class ValueSelectionStrategyImp(ValueSelectionStrategy):

    def choose_value(
            self,
            name: str,
            template: InputTemplate,
            n_samples: int = 1
        ) -> int:
        if n_samples < 1:
            raise ValueError(
                f"Number of samples to sample from {name} must be positive, got {n_samples}"
            )
        
        candidates: List[int] = template.sample(name, n_samples)
        
        if not candidates:
            raise ValueError(f"No candidates returned for {name} from template")
        
        # For now, just take the first one generated
        return candidates[0]

