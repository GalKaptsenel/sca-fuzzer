"""TikTag-v1 (MTE) as a Python object-template: the timing-critical chain is PLANTED (a deep SDIV window
feeding a forced-arch-taken branch; inside the mispredicted body, the tag-checked guess load, a
data-dependent latency gate, and a fixed-set probe), while the rest is CONSTRAINED random fill via holes
(Kind.ALU, non-memory, over registers that EXCLUDE the chain's registers so the random code can't clobber
it or add flag writes before the branch). This shows how much can be randomized around the MTE chain
while preserving the window-edge timing the shrink needs. The probe stays planted — randomizing it
destroys the timing (see LOG.md Exp16/17).

Chain registers (never given to holes): X1,X2 (window), X5 (guess ptr), X9 (branch), X10 (probe addr),
X11,X12,X13 (gate/probe). Hole registers: X0,X3,X4,X6,X7,X8.
"""
from src.aarch64.template import (Template, Hole, Kind, Mem, Imm, specs, flags,
                                  X0, X1, X2, X3, X4, X5, X6, X7, X8, X9, X10, X11, X12, X13, XZR)

HOLE_REGS = [X0, X3, X4, X6, X7, X8]


class MteTiktagPyTmpl(Template):
    def build(self, b):
        b.hole(Hole((6, 12), kind=Kind.ALU, regs=HOLE_REGS))     # random padding BEFORE (constrained)

        b.instruction(specs.STR, XZR, Mem(X5, offset=Imm(0)))    # arch warm+zero the guess granule (offset pinned 0)
        b.instruction(specs.LDR, X12, Mem(X5, offset=Imm(0)))    # warm it to L1 (arch) -> tag check resolves early
        b.instruction(specs.ORR, X10, XZR, Imm(2048))            # probe base -> set 32
        b.instruction(specs.SDIV, X9, X1, X2)                    # window: deep SDIV chain
        for _ in range(20):
            b.instruction(specs.SDIV, X9, X9, X2)
        b.instruction(specs.ORR, X9, X9, Imm(1))                 # force x9 != 0
        b.instruction(specs.SUBS, XZR, X9, Imm(0))               # Z=0 always -> EQ false -> body arch-NOT-taken
        with b.if_(flags.EQ):                                    # body = taken block, reached only speculatively
            b.instruction(specs.LDR, X12, Mem(X5, offset=Imm(0)))  # tag-checked guess load (sealer-decoyed), same line
            b.instruction(specs.AND, X11, X12, XZR)              # zero, but data-dependent on the guess
            for _ in range(8):
                b.instruction(specs.ORR, X10, X10, X11)          # latency gate: probe waits on the guess
            b.instruction(specs.LDR, X13, Mem(X10, offset=Imm(0)))  # probe -> fixed set 32

        b.hole(Hole((6, 12), kind=Kind.ALU, regs=HOLE_REGS))     # random padding AFTER (constrained)
