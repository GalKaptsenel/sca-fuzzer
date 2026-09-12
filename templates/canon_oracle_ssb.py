"""BTB cross-input leftover, fuzzable via the regular branch-target seal (mirrors tcV4br2).

SSB window: a SLOW store address (x4 = input x1, computed through a UDIV+MADD chain) writes 0 to a
location; a FAST load of the SAME location (base x1) bypasses the still-unresolved store and reads the
STALE value. The store address stays unresolved for the whole chain, so the stale-driven BLR target
keeps speculating that long -> the BTB is trained strongly (this is the window length that makes the
cross-input leftover survive into the prober). Architecturally the store lands (flag 0) so the NE guard
is not taken -> the BLR is spec-only (regular sealer seals its target). The stale value dispatches to
one of three targets at distinct cache sets; the canonical leaker allocates BTB->its target and the
prober reads that leftover."""
from src.aarch64.template import (Template, IndirectCall, Mem, Imm, specs, flags,
                                  X0, X1, X3, X4, X5, XZR)


def _fixed_set_load(off):
    def body(f):
        f.instruction(specs.ORR, X3, XZR, Imm(off))
        f.instruction(specs.LDR, X0, Mem(X3, offset=Imm(0)))
    return body


class CanonOracleSsb(Template):
    def build(self, b):
        f1 = b.function(_fixed_set_load(512))     # set 8
        f2 = b.function(_fixed_set_load(1024))    # set 16
        f3 = b.function(_fixed_set_load(2048))    # set 32
        b.instruction(specs.UDIV, X5, X1, X1)              # x5 = 1
        for _ in range(15):
            b.instruction(specs.UDIV, X5, X5, X5)          # x5 = 1, slow (16-deep udiv chain)
        for _ in range(3):
            b.instruction(specs.MADD, X4, X1, X5, XZR)     # x4 = x1*x5 = x1, slow store address
        b.instruction(specs.STR, XZR, Mem(X4, offset=Imm(0)))   # SLOW store 0 -> arch flag 0
        b.instruction(specs.LDR, X0, Mem(X1, offset=Imm(0)))    # FAST load, same loc -> stale (!=0)
        b.instruction(specs.SUBS, XZR, X0, Imm(0))
        with b.if_(flags.NE):                              # spec-taken (stale != 0), arch not-taken
            # dispatch on the STALE load x0 -> the target speculates for the whole store window
            b.call(IndirectCall(targets=[f1, f2, f3], dispatch=True, index_reg=X0))
