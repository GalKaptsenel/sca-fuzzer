"""Ablation: ONLY a fixed SDIV latency chain; everything else -- the conditional branches and the whole
payload (loads, stores, ALU) -- is random. Tests whether the window-widener alone, not feeding any
particular branch, is enough to make random code produce a TikTag tag-check leak. The multi-block hole
lets the generator wire random conditional branches between blocks (the speculation windows), and ANY
holes include MTE-sealable memory accesses."""
from src.aarch64.template import Template, Hole, Kind, specs, X1, X5


class MteSdivOnly(Template):
    def build(self, b):
        b.instruction(specs.SDIV, X5, X1, X1)          # x5 = 1 (x1/x1; div-by-zero yields 0, no fault)
        for _ in range(15):
            b.instruction(specs.SDIV, X5, X5, X5)      # the ONLY fixed content: 16-deep latency chain

        b.hole(Hole(n=(30, 48), n_bb=(4, 6), kind=Kind.ANY))   # random branches + random payload
