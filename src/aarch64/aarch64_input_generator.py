"""
File: AArch64 input generator — per-flag NZCV randomisation of the flags slot.
"""
import numpy as np
from ..input_generator import NumpyRandomInputGenerator
from .aarch64_input_layout import NZCVScheme


class AArch64InputGenerator(NumpyRandomInputGenerator):
    """AArch64-specific input generator with per-flag NZCV randomisation.

    Overrides slot 6 (NZCV register) so each flag occupies bit 0 of its own
    byte (bytes 48-51), giving full byte-granularity taint separability for
    all four flags (N, Z, C, V).  A deterministic auxiliary RNG seeded from
    the same state is used so the override does not disturb the main RNG
    state used for all other registers.
    """

    def _generate_one(self, state: int):
        input_, next_state = super()._generate_one(state)
        nzcv_rng = np.random.default_rng(seed=state ^ 0xDEADBEEFCAFEBABE)
        for i in range(len(input_)):
            input_[i]['gpr'][NZCVScheme.SLOT_IDX] = NZCVScheme.make_random(nzcv_rng)
        return input_, next_state
