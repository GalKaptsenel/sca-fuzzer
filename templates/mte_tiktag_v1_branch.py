"""TikTag-v1 (speculation-shrinkage) MTE gadget mirroring the official compsec-snu/tiktag Gadget 1:
a slow input-dependent divide chain feeds a conditional branch so it resolves late (long window) while
its direction trains the predictor; inside the speculated (mispredicted) path a
tag-checked load is followed, after a gap, by a FIXED separate probe load. A tag mismatch on the
checked load shrinks the window so the probe never issues and its cache line stays absent; a matching
tag keeps the window open and the probe fills its line. The probe's fixed cache set thus diverges
between a genuine (matching) and a decoy (mismatched) tag, while the tag-blind contract trace keeps the
probe present in both -> a speculative-exception violation.

Window is a CONDITIONAL-BRANCH speculation window (contract clause `cond`), not a store-bypass window:
the Cortex-X3 leaks v1 through branch/predictor speculation shrinkage, which a store-bypass gadget does
not exercise. The branch is architecturally not-taken and speculatively taken (trained across the input
batch by cross-input priming), so the checked and probe loads are speculative-only and never retire.

The probe reads a fixed in-region set (distinct from the branch-condition load), so its divergence is a
binary present/absent, as in the official gadget's test load -- not a data-dependent set selection.

Registers: input-seeded x0-x5. Holes outside the branch are architectural (Kind.ALU); the gap inside is
Kind.ALU too (non-memory, matching the official gadget's NOP gap, so it adds no cache traffic).
"""
from src.aarch64.template import Template, Hole, Kind, Mem, Imm, specs, flags, X0, X1, X2, X3, X4, X5, XZR

PROBE_OFF = 2048        # fixed probe address -> cache set 32 (2048 / 64)


class MteTiktagV1Branch(Template):
    def build(self, b):
        b.hole(Hole((2, 6), kind=Kind.ALU))                        # BEFORE (arch, non-mem)

        b.instruction(specs.SDIV, X0, X1, X2)                 # input-dependent slow divide (window opener)
        for _ in range(7):
            b.instruction(specs.SDIV, X0, X0, X2)             # dependent divide chain: branch resolves late
        b.instruction(specs.SUBS, XZR, X0, Imm(0))
        with b.if_(flags.NE):                                 # input-dependent dir trains BPU; late resolve = long window
            b.instruction(specs.LDR, X2, Mem(X3))             # tag-CHECKED guess load (x3 MTE-sealed)
            b.hole(Hole((2, 4), kind=Kind.ALU))                     # gap (non-mem; cycle distance)
            b.instruction(specs.ORR, X4, XZR, Imm(PROBE_OFF)) # fixed probe address
            b.instruction(specs.LDR, X5, Mem(X4))             # PROBE load -> fixed set; present iff reached

        b.hole(Hole((2, 6), kind=Kind.ALU))                        # AFTER (arch, non-mem)
