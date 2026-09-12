"""v4 leftover template with generator-filled (random) target functions -- no hand-specified contents
or registers -- to test that the detector generalizes beyond crafted target sets."""
from src.aarch64.template import (Template, IndirectCall, Mem, Imm, specs, flags,
                                  X0, X1, X4, X5, XZR)


class CanonOracleV4BGeneric(Template):
    def build(self, b):
        f1 = b.function()   # random-filled body, generator's choice of registers/instructions
        f2 = b.function()
        f3 = b.function()
        b.instruction(specs.UDIV, X5, X1, X1)
        for _ in range(15):
            b.instruction(specs.UDIV, X5, X5, X5)
        b.instruction(specs.UDIV, X4, X1, X5)          # slow store address
        b.instruction(specs.STR, XZR, Mem(X4, offset=Imm(0)))
        b.instruction(specs.LDR, X0, Mem(X1, offset=Imm(0)))   # v4 stale load
        b.instruction(specs.SUBS, XZR, X0, Imm(0))
        with b.if_(flags.NE):
            b.call(IndirectCall(targets=[f1, f2, f3], dispatch=True, index_reg=X0))
