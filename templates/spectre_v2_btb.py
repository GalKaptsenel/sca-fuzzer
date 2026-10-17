"""Spectre-v2 (indirect branch) over the BTB -- the Figure-(a) "what priming cannot find" gadget (a target
returns a value, the victim loads from it), generalized to three targets. PURE which-target leak: every
target is canonical and accesses NO memory itself; each just returns a distinct single-bit sandbox offset
(the paper's func0->0, func1->128). The victim loads from the RETURNED value after the call, so WHICH
target the BTB predicts selects the L1D set.

Detected under a CT-SEQ contract: the whole indirect dispatch runs only speculatively (an SSB/v4 window --
a slow store address + a fast stale load -- means architecturally the guard is not taken, so the BLR never
runs non-speculatively). Under CT-SEQ the model has no speculation, so the dispatch and its target are
beyond-contract and the bits that choose the speculative target are UNTAINTED; boosting then varies them,
giving each contract class several members that train the BTB to DIFFERENT targets. A leaker member trains
BTB[blr]->its target; a later contract-equal member mispredicts to it and its distinct L1D set reveals the
leaker's choice -- a strictly cross-input leak standard priming drops and generalized priming catches. No
branch-target sealing, no PAC: the BTB training is a plain speculative which-target mispredict."""
from src.aarch64.template import (Template, IndirectCall, Mem, Imm, specs, flags,
                                  X0, X1, X3, X4, X5, XZR)


def _return_offset(off):
    # set the return value x3 = off (a distinct, single-bit sandbox offset); NO memory access in the target.
    def body(f):
        f.instruction(specs.ORR, X3, XZR, Imm(off))
    return body


class SpectreV2Btb(Template):
    def build(self, b):
        f0 = b.function(_return_offset(512))      # -> set 8
        f1 = b.function(_return_offset(1024))     # -> set 16
        f2 = b.function(_return_offset(2048))     # -> set 32
        b.instruction(specs.UDIV, X5, X1, X1)              # x5 = 1
        for _ in range(15):
            b.instruction(specs.UDIV, X5, X5, X5)          # slow 16-deep udiv chain
        for _ in range(3):
            b.instruction(specs.MADD, X4, X1, X5, XZR)     # x4 = x1*x5 = x1, slow store address
        b.instruction(specs.STR, XZR, Mem(X4, offset=Imm(0)))   # SLOW store 0 -> arch guard 0
        b.instruction(specs.LDR, X0, Mem(X1, offset=Imm(0)))    # FAST load same loc -> stale (!=0) spec
        b.instruction(specs.SUBS, XZR, X0, Imm(0))
        with b.if_(flags.NE):                              # spec-taken only (arch store landed -> x0==0)
            # dispatch on the stale x0 -> the BTB-trained target speculates for the whole store window
            b.call(IndirectCall(targets=[f0, f1, f2], dispatch=True, index_reg=X0))
            b.instruction(specs.LDR, X0, Mem(X3, offset=Imm(0)))  # load FROM the returned offset (which-target)
