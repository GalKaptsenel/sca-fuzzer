from __future__ import annotations
from typing import Any, Dict, Generic, Iterable, List, Mapping, Optional, Tuple, TypeVar, Union
import numpy as np
import random
import math
from itertools import product
import matplotlib.pyplot as plt
import warnings

T = TypeVar("T")  # domain element type

class BayesBase(Generic[T]):
    """
    Generic Bayesian updater over a discrete domain.
    Supports single-variable or Cartesian product (joint) domains.
    """
    def __init__(
        self,
        domain: Union[Iterable[T], List[List[T]]],
        alpha: float = 3.0,
        beta: float = 0.2,
        reward_clip: Optional[Tuple[float, float]] = None
    ) -> None:
        self.alpha: float = float(alpha)
        self.beta: float = float(beta)
        self.reward_clip: Optional[Tuple[float, float]] = reward_clip

        # Detect joint domain
        if isinstance(domain, list) and domain and all(isinstance(d, list) for d in domain):
            # joint domain
            self.is_joint = True
            self.domains: List[List[T]] = domain
            if any(not d for d in domain):
                raise ValueError("All domains must be non-empty lists.")
            self.flat: List[Tuple[T, ...]] = list(product(*self.domains))
        else:
            # single variable domain
            self.is_joint = False
            self.domain: List[T] = list(domain)
            if not self.domain:
                raise ValueError("Domain must contain at least one element.")
            self.flat: List[Tuple[T, ...]] = [(v,) for v in self.domain]

        self._index_map: Mapping[Tuple[T, ...], int] = {t: i for i, t in enumerate(self.flat)}
        self._indices: List[int] = list(range(len(self.flat)))

        self._init_uniform_prior()

    def _init_uniform_prior(self) -> None:
        """Set the prior to a uniform distribution over the domain."""
        n = len(self._indices)
        self.prior: np.ndarray = np.full(n, 1.0 / n, dtype=float)

    def _clip_reward(self, reward: float) -> float:
        if self.reward_clip is not None:
            low, high = self.reward_clip
            return max(min(reward, high), low)
        return reward

    def _sample_index(self, rnd: Optional[random.Random] = None, temperature: float = 1.0) -> int:
        rnd = rnd or random.Random()
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if temperature == 1.0:
            return rnd.choices(self._indices, weights=self.prior, k=1)[0]

        # temperature-scaled softmax
        scaled = np.log(np.clip(self.prior, 1e-12, None)) / temperature
        scaled -= np.max(scaled)
        probs = np.exp(scaled)
        probs /= probs.sum()
        return rnd.choices(self._indices, weights=probs, k=1)[0]

    def _update_index(self, idx: int, reward: float) -> None:
        self._update_indices([idx], [reward])

    def _update_indices(self, indices: List[int], rewards: List[float]) -> None:
        if not indices:
            return
        clipped_rewards = [self._clip_reward(r) for r in rewards]
        log_likelihoods = np.zeros_like(self.prior, dtype=float)
        for idx, r in zip(indices, clipped_rewards):
            log_likelihoods[idx] = self.alpha * r
        max_log = np.max(log_likelihoods)
        exp_shifted = np.exp(log_likelihoods - max_log)
        posterior = self.prior * exp_shifted
        s = float(posterior.sum())
        if s == 0.0 or not math.isfinite(s):
            warnings.warn("Posterior is degenerate or non-finite. Update skipped.")
            return
        posterior /= s
        self.prior = (1.0 - self.beta) * self.prior + self.beta * posterior

    def _unwrap(self, t: Tuple[T, ...]) -> Union[T, Tuple[T, ...]]:
        """Return single element as T if single-variable, else keep tuple."""
        if not isinstance(t, tuple):
            raise TypeError(f"_unwrap expected a tuple, got {type(t)}")
        return t if self.is_joint else t[0]

    def reset(self) -> None:
        """Reset the prior to uniform."""
        self._init_uniform_prior()

    def entropy(self) -> float:
        """Shannon entropy of the current distribution."""
        p = self.prior[self.prior > 0]
        return float(-np.sum(p * np.log(p)))

    def top_k(self, k: int = 5) -> List[Union[T, Tuple[T, ...]]]:
        k = min(k, len(self.flat))
        idxs = np.argsort(self.prior)[::-1][:k]
        return [self._unwrap(self.flat[i]) for i in idxs]

    def probs(self) -> Dict[str, float]:
        return {str(self._unwrap(t)): float(p) for t, p in zip(self.flat, self.prior)}

    def most_likely(self) -> Union[T, Tuple[T, ...]]:
        idx = int(np.argmax(self.prior))
        return self._unwrap(self.flat[idx])

    def plot_prior(self, top_k: Optional[int] = None, log_scale: bool = False, annotate: bool = True) -> None:
        labels = [str(self._unwrap(t)) for t in self.flat]
        probs = self.prior
        if top_k is not None:
            k = min(top_k, len(probs))
            idxs = np.argsort(self.prior)[::-1][:k]
            probs = self.prior[idxs]
            labels = [labels[i] for i in idxs]
        most_likely_idx = int(np.argmax(probs))
        colors = ['orange' if i == most_likely_idx else 'steelblue' for i in range(len(probs))]
        width = max(6, len(probs) * 0.5)
        plt.figure(figsize=(width, 4))
        bars = plt.bar(range(len(probs)), probs, tick_label=labels, color=colors)
        plt.xticks(rotation=45, ha='right')
        plt.ylabel("Probability")
        plt.title("Bayesian Prior Distribution")
        if log_scale:
            plt.yscale("log")
            plt.ylabel("Probability (log scale)")
        if annotate:
            for bar, prob in zip(bars, probs):
                plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{prob:.3f}", ha='center', va='bottom', fontsize=8)
        plt.tight_layout()
        plt.show()


class DiscreteBayes(BayesBase[T]):
    """Discrete Bayesian updater for a single variable."""
    def sample(self, rnd: Optional[random.Random] = None, temperature: float = 1.0) -> T:
        idx = self._sample_index(rnd, temperature)
        return self._unwrap(self.flat[idx])

    def update(self, value: T, reward: float) -> None:
        if (value,) not in self._index_map:
            warnings.warn(f"Value {value} not in domain, update skipped.")
            return
        idx = self._index_map[(value,)]
        self._update_index(idx, reward)

    def update_batch(self, values: List[T], rewards: List[float]) -> None:
        if len(values) != len(rewards):
            raise ValueError("values and rewards must have the same length")
        indices = [self._index_map[(v,)] for v in values if (v,) in self._index_map]
        filtered_rewards = [r for v, r in zip(values, rewards) if (v,) in self._index_map]
        if len(indices) < len(values):
            warnings.warn(f"{len(values) - len(indices)} values ignored in batch update (not in domain).")
        self._update_indices(indices, filtered_rewards)


class JointDiscreteBayes(BayesBase[T]):
    """Discrete Bayesian updater over Cartesian-product domains."""
    def sample(self, rnd: Optional[random.Random] = None, temperature: float = 1.0) -> Tuple[T, ...]:
        idx = self._sample_index(rnd, temperature)
        return self.flat[idx]

    def update(self, value: Tuple[T, ...], reward: float) -> None:
        if value not in self._index_map:
            warnings.warn(f"Value {value} not in domain, update skipped.")
            return
        idx = self._index_map[value]
        self._update_index(idx, reward)

    def update_batch(self, values: List[Tuple[T, ...]], rewards: List[float]) -> None:
        if len(values) != len(rewards):
            raise ValueError("values and rewards must have the same length")
        indices = [self._index_map[v] for v in values if v in self._index_map]
        filtered_rewards = [r for v, r in zip(values, rewards) if v in self._index_map]
        if len(indices) < len(values):
            warnings.warn(f"{len(values) - len(indices)} values ignored in batch update (not in domain).")
        self._update_indices(indices, filtered_rewards)

