"""Simplest v4-based BTB cross-input leftover template. Dispatch table + the regular canonicality/
alignment branch-target seal. Uses only udiv/orr/str/ldr/subs + IndirectCall (no madd/movz/movk).

v4 window: x5 = slow 1 (udiv chain); x4 = x1 / x5 = x1 but data-dependent on the slow chain -> SLOW
store address. STR 0 at x29+(x1&mask); a FAST load from the same x29+(x1&mask) bypasses the unresolved
store and reads the STALE value. Arch: store lands 0 -> NE guard not taken -> BLR spec-only (sealable).
Spec: stale != 0 -> BLR reached; stale dispatches to f1/f2/f3 (sets 8/16/32). The canonical leaker
trains BTB->its target for the whole slow-store window; the prober reads that leftover."""
from src.aarch64.template import (Template, IndirectCall, Mem, Imm, specs, flags,
                                  X0, X1, X3, X4, X5, XZR)


def _fixed_set_load(off):
    def body(f):
        f.instruction(specs.ORR, X3, XZR, Imm(off))
        f.instruction(specs.LDR, X0, Mem(X3, offset=Imm(0)))
    return body


class CanonOracleV4(Template):
    def build(self, b):
        f1 = b.function(_fixed_set_load(512))     # set 8
        f2 = b.function(_fixed_set_load(1024))    # set 16
        f3 = b.function(_fixed_set_load(2048))    # set 32
        b.instruction(specs.UDIV, X5, X1, X1)              # x5 = 1
        for _ in range(15):
            b.instruction(specs.UDIV, X5, X5, X5)          # x5 = 1, slow
        b.instruction(specs.UDIV, X4, X1, X5)              # x4 = x1/1 = x1, slow store address
        b.instruction(specs.STR, XZR, Mem(X4, offset=Imm(0)))   # SLOW store 0 -> arch flag 0
        b.instruction(specs.LDR, X0, Mem(X1, offset=Imm(0)))    # FAST load, same loc -> stale (!=0)
        b.instruction(specs.SUBS, XZR, X0, Imm(0))
        with b.if_(flags.NE):                              # spec-taken (stale != 0), arch not-taken
            b.call(IndirectCall(targets=[f1, f2, f3], dispatch=True, index_reg=X0))
