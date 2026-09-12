"""v4 leftover template with target sets chosen OFF the dispatch/store baselines (16/32/48)."""
from src.aarch64.template import (Template, IndirectCall, Mem, Imm, specs, flags,
                                  X0, X1, X3, X4, X5, XZR)
def _const_load(off):
    def body(f):
        f.instruction(specs.ORR, X3, XZR, Imm(off))
        f.instruction(specs.LDR, X0, Mem(X3, offset=Imm(0)))
    return body
class CanonOracleV4B(Template):
    def build(self, b):
        f1=b.function(_const_load(1024))  # set16
        f2=b.function(_const_load(2048))  # set32
        f3=b.function(_const_load(3072))  # set48
        b.instruction(specs.UDIV, X5, X1, X1)
        for _ in range(15): b.instruction(specs.UDIV, X5, X5, X5)
        b.instruction(specs.UDIV, X4, X1, X5)
        b.instruction(specs.STR, XZR, Mem(X4, offset=Imm(0)))
        b.instruction(specs.LDR, X0, Mem(X1, offset=Imm(0)))
        b.instruction(specs.SUBS, XZR, X0, Imm(0))
        with b.if_(flags.NE):
            b.call(IndirectCall(targets=[f1,f2,f3], dispatch=True, index_reg=X0))
