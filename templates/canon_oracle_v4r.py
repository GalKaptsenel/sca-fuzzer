"""canon_oracle_v4 with SHORT, RANDOM-memory-access target functions (more general than fixed-set
loads). Each of the three dispatch targets is a short random-fill body containing a memory access, so
the leaked cache set is whatever that body happens to touch -- not a hand-picked constant. Detection is
black-box (the scan compares htraces), so it does not depend on the set identities, only on the three
targets touching distinguishable sets."""
from src.aarch64.template import (Template, IndirectCall, Mem, Imm, specs, flags, Hole, Kind,
                                  X0, X1, X4, X5, XZR)


class CanonOracleV4R(Template):
    def build(self, b):
        # three short random-memory callees (a load-bearing body; the generator wires the RET)
        f1 = b.function(lambda f: f.hole(Hole(2, kind=Kind.LOAD)))
        f2 = b.function(lambda f: f.hole(Hole(2, kind=Kind.LOAD)))
        f3 = b.function(lambda f: f.hole(Hole(2, kind=Kind.LOAD)))
        b.instruction(specs.UDIV, X5, X1, X1)
        for _ in range(15):
            b.instruction(specs.UDIV, X5, X5, X5)
        b.instruction(specs.UDIV, X4, X1, X5)              # x4 = x1, slow store address
        b.instruction(specs.STR, XZR, Mem(X4, offset=Imm(0)))
        b.instruction(specs.LDR, X0, Mem(X1, offset=Imm(0)))    # fast load -> stale
        b.instruction(specs.SUBS, XZR, X0, Imm(0))
        with b.if_(flags.NE):
            b.call(IndirectCall(targets=[f1, f2, f3], dispatch=True, index_reg=X0))
