import copy
import random
from typing import Dict, Any, Callable, Optional, Union

from .input_generator import ValueGenerator



class InputTemplate:
    def __init__(self):
        # key ->   {
        #               "gen": ValueGenerator,
        #               "params": Params, 
        #               "fixed": fixed_value
        #           }
        self.inputs: Union[str, int] = {}

    def register_unknown(self, key: Union[str, int]):
        self.inputs[key] = {"gen": generator, "params": params or {}, "fixed": None}

    def set_concrete(self, key: Union[str, int], value: int):
        if key not in self.inputs:
            raise ValueError(f"Unknown input key {key}")

        info = self.inputs[key]
        if info["fixed"] is not None:
            raise ValueError(f"input {key} is already concrete value (previously set to {info['fixed']})")

        info["fixed"] = value

    def is_concrete(self, key: Union[str, int]) -> bool:
        if key not in self.inputs:
            raise ValueError(f"Unknown input key {key}")
        return self.input[key]["fixed"] is not None

    def get_concrete(self, key: Union[str, int]) -> int:
        if not self.is_concrete(key):
            raise ValueError(f"input key {key} is not concrete!")

        return self.input[key]["fixed"]


    def sample(self, key: Union[str, int], n_samples: int, rnd: Optional[random.Random] = None) -> List[int]:
        rnd = rnd or random.Random()
        if not isinstance(rnd, random.Random):
            raise ValueError(f"rnd should be of type random.Random, got {type(rnd)}\n")

        if not isinstance(key, (str, int)):
            raise ValueError(f"key should be of type Union[str, int], got {type(key)}\n")

        if self.is_concrete(key):
            return [self.inputs[key]["fixed"]] * n_samples

        return [
                info = self.inputs[key]
                info["gen"](info["params"], rnd)
                for _ in rane(n_samples)
        ]


    def update_posteriors(self, reward, sampled_inputs):
        for key, info in self.inputs.items():
            current_params = info["params"]
            dist.update(reward, sampled_inputs[key])
            info["params"] = new_params
    
    def clone(self) -> InputTemplate:
        return copy.deepcopy(self)

