"""
File: AArch64 input generator — per-flag NZCV randomisation of the flags slot.
"""
import numpy as np
from typing import List, Optional, Tuple
from ..input_generator import NumpyRandomInputGenerator
from ..interfaces import Input
from ..config import CONF
from .aarch64_input_layout import NZCVScheme
from .seal.pac import pac_enabled
from . import aarch64_executor_input_encoder as wire


class AArch64InputGenerator(NumpyRandomInputGenerator):
    """AArch64-specific input generator with per-flag NZCV randomisation.

    Overrides slot 6 (NZCV register) so each flag occupies bit 0 of its own
    byte (bytes 48-51), giving full byte-granularity taint separability for
    all four flags (N, Z, C, V).  A deterministic auxiliary RNG seeded from
    the same state is used so the override does not disturb the main RNG
    state used for all other registers.
    """

    _shared_pac_keys: Optional[Tuple[int, ...]] = None

    def _generate_one(self, state: int):
        input_, next_state = super()._generate_one(state)
        nzcv_rng = np.random.default_rng(seed=state ^ 0xDEADBEEFCAFEBABE)
        for i in range(len(input_)):
            input_[i]['gpr'][NZCVScheme.SLOT_IDX] = NZCVScheme.make_random(nzcv_rng)
        input_.pac_keys = self._pac_keys_for(state)
        return input_, next_state

    def _pac_keys_for(self, state: int) -> Optional[Tuple[int, ...]]:
        """The input's PAC keys (None without PAC): drawn from its own seed, or (shared mode) once from
        the first seed this generator produced from."""
        if not pac_enabled():
            return None
        if CONF.pac_keys_per_input:
            return _draw_pac_keys(state)
        if self._shared_pac_keys is None:
            self._shared_pac_keys = _draw_pac_keys(state)
        return self._shared_pac_keys

    def extend_equivalence_classes(self, inputs: List[Input], taints) -> List[Input]:
        """Boosted inputs run under their base input's PAC keys."""
        new_inputs = super().extend_equivalence_classes(inputs, taints)
        for base, new in zip(inputs, new_inputs):
            new.pac_keys = base.pac_keys
        return new_inputs

    def _load_one(self, input_path: str) -> Input:
        with open(input_path, "rb") as f:
            input_ = wire.deserialize(f.read()).input_
        if pac_enabled() and input_.pac_keys is None:
            raise ValueError(f"{input_path}: a PAC campaign input must carry its PAC keys")
        return input_


_PAC_KEYS_SALT = 0x5041435F4B455953   # "PAC_KEYS"


def _draw_pac_keys(seed: int) -> Tuple[int, ...]:
    """10 PAC key words (apia, apib, apda, apdb, apga as {lo, hi}) from `seed`."""
    rng = np.random.default_rng(seed=seed ^ _PAC_KEYS_SALT)
    return tuple(int(w) for w in rng.integers(0, np.iinfo(np.uint64).max, size=10, dtype=np.uint64,
                                              endpoint=True))
