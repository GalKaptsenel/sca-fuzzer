"""canon_oracle_v4 with short targets that load from a RANDOM but CONSTANT (code-intrinsic) offset.
General (offsets are not hand-picked) yet detectable: because the address is an immediate, the leaked
cache set is a property of the target's CODE, so it transfers when the prober speculatively runs the
leaker's target (which happens with the prober's own registers -- register-addressed loads would not
transfer). The 3 offsets map to 3 distinct sets."""
import random
from src.aarch64.template import (Template, IndirectCall, Mem, Imm, specs, flags,
                                  X0, X1, X3, X4, X5, XZR)


def _const_load(off):
    def body(f):
        f.instruction(specs.ORR, X3, XZR, Imm(off))          # x3 = off (valid logical immediate)
        f.instruction(specs.LDR, X0, Mem(X3, offset=Imm(0))) # load at x29+(off&mask) -> set off>>6
    return body


class CanonOracleV4RC(Template):
    def build(self, b):
        bits = random.sample([9, 10, 11, 12], 3)            # distinct single-bit offsets -> distinct sets
        f1, f2, f3 = (b.function(_const_load(1 << k)) for k in bits)
        b.instruction(specs.UDIV, X5, X1, X1)
        for _ in range(15):
            b.instruction(specs.UDIV, X5, X5, X5)
        b.instruction(specs.UDIV, X4, X1, X5)
        b.instruction(specs.STR, XZR, Mem(X4, offset=Imm(0)))
        b.instruction(specs.LDR, X0, Mem(X1, offset=Imm(0)))
        b.instruction(specs.SUBS, XZR, X0, Imm(0))
        with b.if_(flags.NE):
            b.call(IndirectCall(targets=[f1, f2, f3], dispatch=True, index_reg=X0))
