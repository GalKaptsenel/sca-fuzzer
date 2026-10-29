"""Easy-to-fuzz canonicality leftover template. ONE sealed indirect target so ANY two speculative
inputs share the SAME BTB entry -> a leaker/prober pair is trivial to generate:
  * v4/SSB window (SSBS=1 + a slow UDIV-chain store) makes the BLR region speculative every run;
  * the flag comes from a slow pointer-chase into x1 -> for random inputs x1 != 0 almost always ->
    the conditional is NOT-taken -> the BLR is SPECULATIVE-only (so branch-target sealing applies and a
    non-canonical decoy faults harmlessly instead of crashing);
  * a SINGLE indirect target (no dispatch) -> the leaker trains BTB->target when canonical, faults and
    trains nothing when non-canonical, and the prober (same BLR PC) reads that one entry.
The genuine variant seals the target canonical (BLR completes, trains); the decoy seals it non-canonical
(BLR faults, trains nothing). The leftover: prober's target-set fires iff its predecessor was canonical."""
from src.aarch64.template import (Template, IndirectCall, Mem, Imm, specs, flags,
                                  X0, X1, X2, X4, X8, XZR)


class CanonOracleEasy(Template):
    def build(self, b):
        target = b.function(lambda f: f.instruction(specs.LDR, X0, Mem(X4)))  # one target -> set from X4
        # v4 window: slow store address via a UDIV chain -> wide speculation window every run
        for _ in range(12):
            b.instruction(specs.UDIV, X8, X1, X1)
        b.instruction(specs.STR, X2, Mem(X8))
        # flag: a slow pointer-chase into x1 (random -> != 0 -> conditional not-taken -> BLR speculative)
        b.instruction(specs.LDR, X0, Mem(X1))
        b.instruction(specs.LDR, X1, Mem(X0))
        b.instruction(specs.LDR, X0, Mem(X1))
        b.instruction(specs.LDR, X1, Mem(X0))
        b.instruction(specs.SUBS, XZR, X1, Imm(0))
        with b.if_(flags.EQ):
            b.call(IndirectCall(targets=[target]))   # SINGLE indirect target -> shared BTB entry
