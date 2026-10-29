"""BTB cross-input leftover template, easy to fuzz. Three indirect targets whose cache sets are FIXED and
DISTINCT (independent of the input) so the cross-input BTB leftover always surfaces:
  * v4/SSB window (SSBS=1 + slow UDIV-chain store) -> the BLR region speculates every run;
  * flag from a slow pointer-chase into x1 -> random x1 != 0 -> conditional NOT-taken -> BLR SPECULATIVE
    (branch-target sealing applies; a non-canonical decoy faults harmlessly instead of crashing);
  * a 3-way dispatch (x2&3) over f1/f2/f3; each target loads a FIXED sandbox offset in a DISTINCT set:
    f1 -> set 8 (off 512), f2 -> set 16 (off 1024), f3 -> set 32 (off 2048); single-bit offs = valid ORR imm.
The LEAKER (a speculative input dispatching to, say, f2) trains BTB->f2 when canonical; the PROBER (another
speculative input) then speculatively runs f2 via that BTB entry -> set 24 fires -- distinct from the
prober's own target set and from f1 -- iff its predecessor was canonical. That distinct set is the
cross-input signal our leaker-finder priming isolates."""
from src.aarch64.template import (Template, IndirectCall, Mem, Imm, specs, flags,
                                  X0, X1, X2, X8, X12, XZR)


def _fixed_set_load(off):
    # MOV x12,#off ; LDR x0,[x12]  -> the sandbox pass clamps x12 to base+(off&0x1fff) -> set (off>>6)&63.
    def body(f):
        f.instruction(specs.ORR, X12, XZR, Imm(off))
        f.instruction(specs.LDR, X0, Mem(X12))
    return body


class CanonOracleBtb(Template):
    def build(self, b):
        f1 = b.function(_fixed_set_load(512))    # set 8
        f2 = b.function(_fixed_set_load(1024))   # set 16
        f3 = b.function(_fixed_set_load(2048))   # set 32
        for _ in range(12):
            b.instruction(specs.UDIV, X8, X1, X1)
        b.instruction(specs.STR, X2, Mem(X8))
        b.instruction(specs.LDR, X0, Mem(X1))
        b.instruction(specs.LDR, X1, Mem(X0))
        b.instruction(specs.LDR, X0, Mem(X1))
        b.instruction(specs.LDR, X1, Mem(X0))
        b.instruction(specs.SUBS, XZR, X1, Imm(0))
        with b.if_(flags.EQ):
            b.call(IndirectCall(targets=[f1, f2, f3], dispatch=True, index_reg=X2))
