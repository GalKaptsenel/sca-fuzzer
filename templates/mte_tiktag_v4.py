"""TikTag in a pure Spectre-v4 (store-bypass) window with NO conditional branch, so no branch-predictor
training is needed. A slow store's address stays unresolved while a fast load of the same cell bypasses
it; the memory-order violation squashes and re-executes everything after that load, so the instructions
after it run speculatively (contract clause `bpas`). Right there -- no branch -- a tag-checked load
reads through an MTE-sealed base (x3); its loaded value selects a cache set and a dependent load reads
that set. The speculative-exception sealer perturbs only speculative accesses, so it decoys exactly this
checked load: a genuine (matching) tag forwards the value and the dependent load fills its set, while a
decoy (mismatching) tag suppresses the forward/retire so the dependent load never issues and the set
stays empty. The genuine and decoy hardware traces diverge at that downstream set; the tag-blind
contract trace does not -- the leak -- and it needs no training, loop, or branch misprediction.

Registers: input-seeded x0-x5. Holes before/after are architectural (Kind.ALU); the hole among the
speculative loads is Kind.ANY (speculative-only).
"""
from src.aarch64.template import Template, Hole, Kind, Mem, Imm, specs, flags, \
    X0, X1, X2, X3, X4, X5, XZR

SET_MASK = 0xFC0        # loaded-value bits [11:6] -> a 64-byte-line cache set within the region
LDNUM = 8               # repeated checked loads deepen the speculative tag-check pressure


class MteTiktagV4(Template):
    def build(self, b):
        b.hole(Hole((2, 6), kind=Kind.ALU))                      # BEFORE (architectural, non-memory)

        b.instruction(specs.LDR, X5, Mem(X1, offset=Imm(0)))     # warm the bypass cell -> fast bypass load
        b.instruction(specs.UDIV, X5, X1, X1)                    # x5 = 1
        for _ in range(15):
            b.instruction(specs.UDIV, X5, X5, X5)                # slow udiv chain -> store address late
        for _ in range(3):
            b.instruction(specs.MADD, X4, X1, X5, XZR)           # x4 = x1 (slow store address, aliases load)
        b.instruction(specs.STR, XZR, Mem(X4, offset=Imm(0)))    # SLOW store 0 to [x1]
        b.instruction(specs.LDR, X0, Mem(X1, offset=Imm(0)))     # FAST load bypasses -> reads stale (!=0)
        b.instruction(specs.SUBS, XZR, X0, Imm(0))
        with b.if_(flags.NE):                                    # stale != 0 -> spec-taken, arch not-taken
            for _ in range(LDNUM):                               #   (data-driven by the bypass, no training)
                b.instruction(specs.LDR, X2, Mem(X3))            # tag-CHECKED spec load (MTE-sealed base x3)
            b.instruction(specs.AND, X4, X2, Imm(SET_MASK))      # loaded value selects a probe set
            b.instruction(specs.LDR, X0, Mem(X4))                # DEPENDENT probe (sandbox-clamped) -> set
            b.hole(Hole((1, 3), kind=Kind.ANY))                  # speculative-only filler

        b.hole(Hole((2, 6), kind=Kind.ALU))                      # AFTER (architectural, non-memory)
