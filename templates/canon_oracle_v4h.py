"""canon_oracle_v4 + random-fill holes before and after the indirect branch (robustness / realism test).
Holes are constrained to x2/x3 so they add surrounding code without clobbering the v4 chain
(x0=stale, x1=addr, x4=slow store addr, x5=slow); x3 is re-loaded inside each target anyway."""
from src.aarch64.template import (Template, IndirectCall, Mem, Imm, specs, flags, Hole, Kind,
                                  X0, X1, X2, X3, X4, X5, XZR)


def _fixed_set_load(off):
    def body(f):
        f.instruction(specs.ORR, X3, XZR, Imm(off))
        f.instruction(specs.LDR, X0, Mem(X3, offset=Imm(0)))
    return body


class CanonOracleV4H(Template):
    def build(self, b):
        f1 = b.function(_fixed_set_load(512))
        f2 = b.function(_fixed_set_load(1024))
        f3 = b.function(_fixed_set_load(2048))
        b.instruction(specs.UDIV, X5, X1, X1)
        for _ in range(15):
            b.instruction(specs.UDIV, X5, X5, X5)
        b.instruction(specs.UDIV, X4, X1, X5)
        b.instruction(specs.STR, XZR, Mem(X4, offset=Imm(0)))
        b.instruction(specs.LDR, X0, Mem(X1, offset=Imm(0)))
        b.instruction(specs.SUBS, XZR, X0, Imm(0))
        with b.if_(flags.NE):
            b.hole(Hole(3, kind=Kind.ALU, regs=[X2, X3]))          # random code BEFORE the indirect branch
            b.call(IndirectCall(targets=[f1, f2, f3], dispatch=True, index_reg=X0))
            b.hole(Hole(3, kind=Kind.ALU, regs=[X2, X3]))          # random code AFTER the indirect branch
