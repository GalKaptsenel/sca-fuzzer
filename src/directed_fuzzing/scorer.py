from abc import ABC, abstractmethod
from collections import Counter
from typing import Tuple

import math
import hashlib

from .microarch import MicroarchState

class ScorerInterface(ABC):
    @abstractmethod
    def __call__(self, state: MicroarchState) -> float:
        pass

class NoveltyScorer(ScorerInterface):
    def __init__(self, decay: float = 0.99, min_reward: float = 0.01, decay_interval: int = 100):
        self._seen_states = Counter()
        self._decay = decay
        self._min_reward = min_reward
        self._decay_interval = decay_interval
        self._calls = 0

    @staticmethod
    def _state_signature(state: MicroarchState) -> Any:
        return state.snapshot()
#        snapshot_bytes = bytes(state.snapshot())
#        return int(hashlib.blake2b(snapshot_bytes, digest_size=8).hexdigest(), 16)

    def __call__(self, state: MicroarchState) -> float:
        self._calls += 1

        # decay all previous counts
        if self._calls % self._decay_interval == 0:
            for sig in list(self._seen_states.keys()):
                self._seen_states[sig] *= self._decay
                if self._seen_states[sig] < 1e-6:
                    del self._seen_states[sig]  # remove almost-zero entries

        sig = self._state_signature(state)
        self._seen_states[sig] += 1

        # Reward inversely proportional to how many times visited the state
        reward = 1.0 / max(self._seen_states[sig], 1.0)
        reward = max(reward, self._min_reward)
        return reward

