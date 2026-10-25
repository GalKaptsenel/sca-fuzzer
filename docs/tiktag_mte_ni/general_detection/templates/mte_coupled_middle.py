"""Coupled middle step: fix ONLY the leak-enabling coupling -- a slow chain that feeds the branch so the
window is wide (a late-resolving condition) -- and randomize everything else. The window is the proven
store-bypass form: a slow UDIV chain delays a store's address while a fast load of the same location
bypasses it and reads the stale value, so the branch is speculatively taken while architecturally not,
and the `if_` body runs speculative-only. Inside that window the payload is a RANDOM hole that includes a
sealable load (the leak site); the prologue, epilogue, and the rest of the window body are random too.

This is the structure a FLOATING slow chain lacks (ablation: chain not feeding the branch finds nothing)
and that blind random essentially never generates: slow-chain -> branch -> sealed load in the shadow."""
from src.aarch64.template import Template, Hole, Kind, specs, flags, Mem, Imm, X0, X1, X4, X5, XZR


class MteCoupledMiddle(Template):
    def build(self, b):
        b.hole(Hole((2, 6), kind=Kind.ALU))                      # random arch prologue (non-mem)

        b.instruction(specs.UDIV, X5, X1, X1)                    # x5 = 1
        for _ in range(15):
            b.instruction(specs.UDIV, X5, X5, X5)                # slow chain -> wide window
        for _ in range(3):
            b.instruction(specs.MADD, X4, X1, X5, XZR)           # x4 = x1, dependent on the slow chain
        b.instruction(specs.STR, XZR, Mem(X4, offset=Imm(0)))    # SLOW store (addr waits on the chain)
        b.instruction(specs.LDR, X0, Mem(X1, offset=Imm(0)))     # FAST load, same loc -> stale (!=0)
        b.instruction(specs.SUBS, XZR, X0, Imm(0))
        with b.if_(flags.NE):                                    # spec-taken, arch-not-taken window
            b.hole(Hole((1, 3), kind=Kind.ANY))                  # random spec payload before the load
            b.hole(Hole(1, kind=Kind.LOAD))                      # random SEALED load in the shadow (leak)
            b.hole(Hole((1, 3), kind=Kind.ANY))                  # random spec payload after the load

        b.hole(Hole((2, 6), kind=Kind.ALU))                      # random arch epilogue (non-mem)
