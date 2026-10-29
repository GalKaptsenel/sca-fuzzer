"""3-class speculative canonicality oracle -- slow-but-decoupled dispatch.

The branch FLAG comes from a long slow pointer-chase into x1 (4 misses) -> wide window. The DISPATCH
index (pinned to x2) comes from a SHORTER slow load into x2 (one miss) -> it resolves *during* the flag's
window, so the speculative BLR can still reach f2/f3 and do its work, while the branch itself is still
mispredicting. Decoupled timing: dispatch ready mid-window, flag resolves at the end.

  training: flag=0 (TAKEN), dispatch=0 -> f2 (trains BTB)
  class2  : flag!=0 (not-taken), dispatch=0 -> speculative f2 (canonical vs non-canonical seal)
  class3  : flag!=0 (not-taken), dispatch=1 -> f3 (step-3 readout)
f2 loads X4 (set 24), f3 loads X5 (set 40).
"""
from src.aarch64.template import (Template, IndirectCall, Mem, Imm, specs, flags,
                                  X0, X1, X2, X3, X4, X5, XZR)


class CanonOracle3(Template):
    def build(self, b):
        f2 = b.function(lambda f: f.instruction(specs.LDR, X0, Mem(X4)))   # victim -> set from X4
        f3 = b.function(lambda f: f.instruction(specs.LDR, X0, Mem(X5)))   # victim -> set from X5
        fT = b.function(lambda f: f.instruction(specs.LDR, X0, Mem(X3)))   # single training target -> X3
        # dispatch index: ONE slow (missing) load into x2 -> ready mid-window
        b.instruction(specs.LDR, X2, Mem(X3))
        # flag: LONGER slow pointer-chase into x1 -> resolves later -> the window
        b.instruction(specs.LDR, X0, Mem(X1))
        b.instruction(specs.LDR, X1, Mem(X0))
        b.instruction(specs.LDR, X0, Mem(X1))
        b.instruction(specs.LDR, X1, Mem(X0))
        b.instruction(specs.SUBS, XZR, X1, Imm(0))     # Z = (x1 == 0) -- slow
        with b.if_(flags.EQ):
            b.call(IndirectCall(targets=[f2, f3, fT], dispatch=True, index_reg=X2))
