"""TikTag in a return-stack-buffer (Spectre-RSB / ret2spec) window -- a CONTROL-FLOW speculation window
(like a branch, where the MTE tag-check shrink lives) that needs NO branch-direction training, because
the misprediction is driven by RSB overflow, not a predictor's learned direction.

A call chain deeper than the hardware RSB overflows it, so the deep returns pop STALE addresses and the
frontend speculatively runs at an earlier call's return site until the architectural return squashes it.
Each function on the chain places, right after its call (its return site), a tag-checked load through an
MTE-sealed base (x3) whose loaded value selects a cache set, then a dependent load of that set. The
speculative-exception sealer perturbs only speculative accesses, so it decoys exactly these return-site loads
(the chain is deeper than the RSB, so their return sites are the mispredicted targets). A genuine
(matching) tag forwards the value and the dependent load fills its set; a decoy (mismatching) tag gates
it and the set stays empty -- the genuine/decoy htraces diverge while the tag-blind contract trace does
not. No branch, no training: the RSB overflow supplies the control-flow window.

The chain depth exceeds any plausible RSB depth, and every return site carries the gadget, so the
mispredicted return lands on a sealed gadget wherever the true RSB depth falls.
"""
from src.aarch64.template import Template, Hole, Kind, Mem, Imm, specs, X0, X2, X3, X4

SET_MASK = 0xFC0        # loaded-value bits [11:6] -> a 64-byte-line cache set within the region
DEPTH = 24              # call-chain depth, deeper than any plausible hardware RSB


def _gadget(f):
    # placed at a return site: speculative under the RSB misprediction window
    f.instruction(specs.LDR, X2, Mem(X3))               # tag-CHECKED load (MTE-sealed base x3)
    f.instruction(specs.AND, X4, X2, Imm(SET_MASK))     # loaded value selects a probe set
    f.instruction(specs.LDR, X0, Mem(X4))               # DEPENDENT probe (sandbox-clamped) -> set


def _make_body(depth):
    def body(f):
        if depth > 1:
            child = f.function(_make_body(depth - 1))   # declare the next, deeper function
            f.call(specs_directcall(child))             # call it; code below is the return site
            _gadget(f)                                  # gadget at this return site (speculative on RSB miss)
    return body


# DirectCall is imported lazily to keep the template's import list minimal.
def specs_directcall(child):
    from src.aarch64.template import DirectCall
    return DirectCall(target=child)


class MteTiktagRsb(Template):
    def build(self, b):
        b.hole(Hole((2, 6), kind=Kind.ALU))                 # BEFORE (architectural, non-memory)
        child = b.function(_make_body(DEPTH - 1))           # head of the deep chain
        b.call(specs_directcall(child))
        _gadget(b)                                          # top return site also carries the gadget
        b.hole(Hole((2, 6), kind=Kind.ALU))                 # AFTER (architectural, non-memory)
