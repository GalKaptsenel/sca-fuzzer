from .saturating_bp import SaturatingCounterBPCommon


class TwoBitBP(SaturatingCounterBPCommon):
    def __init__(self, num_sets: int = 1024, assoc: int = 1):
        super().__init__(counter_bit_width=2, num_sets=num_sets, assoc=assoc)
