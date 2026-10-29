"""TikTag-v1 (speculation-shrink) MTE gadget: a tag-checked speculative load whose check gates a
DEPENDENT downstream access, so the tag result is observed at the downstream cache set, not the
checked load itself.

Store-bypass window (Spectre-v4/SSB): a slow store's address stays unresolved while a fast load of the
same location bypasses it and reads the STALE (non-zero) value, so the branch is SPECULATIVELY taken
while ARCHITECTURALLY not taken. Inside that spec-only branch a load reads through a regularly-
sandboxed, input-derived, MTE-sealed base (x3). Its loaded value selects a cache set, and a dependent
load reads that set (the sandbox clamp lands the selector in the region). On a genuine (matching) tag the checked load retires
into the window and the dependent load fills its set. On a decoy (mismatched) tag the tag check shrinks
the window at the checked load, so the dependent load never issues and its set stays empty. The
genuine/decoy htraces diverge at that downstream set while the tag-blind contract trace does not.

The checked load and the dependent load are both speculative-only, so decoy tag mismatches never retire
and cannot fault. Branch-predictor training comes from the input batch: cross-input priming runs the
window over the class sequence, training it with the matching-tag baseline before the mismatched decoy.

Registers: only the input-seeded x0-x5. Holes before and after the template are architectural (Kind.ALU
only); the holes inside the if are speculative-only (Kind.ANY).
"""
from src.aarch64.template import Template, Hole, Kind, Mem, Imm, specs, flags, X0, X1, X2, X3, X4, X5, XZR

SET_MASK = 0xFC0        # loaded-value bits [11:6] -> a 64-byte-line cache set within the region


class MteTiktagV1(Template):
    def build(self, b):
        b.hole(Hole((2, 6), kind=Kind.ALU))                      # BEFORE the template (arch, non-mem)

        b.instruction(specs.UDIV, X5, X1, X1)                    # x5 = 1
        for _ in range(15):
            b.instruction(specs.UDIV, X5, X5, X5)                # slow udiv chain (long SSB window)
        for _ in range(3):
            b.instruction(specs.MADD, X4, X1, X5, XZR)           # x4 = x1 (slow store address)
        b.instruction(specs.STR, XZR, Mem(X4, offset=Imm(0)))    # SLOW store 0 -> arch flag 0
        b.instruction(specs.LDR, X0, Mem(X1, offset=Imm(0)))     # FAST load, same loc -> stale (!=0)
        b.instruction(specs.SUBS, XZR, X0, Imm(0))
        with b.if_(flags.NE):                                    # spec-taken (stale != 0), arch not
            b.hole(Hole((1, 3), kind=Kind.ANY))                  # within if, before the check (spec-only)
            b.instruction(specs.LDR, X2, Mem(X3))                # tag-CHECKED spec load (MTE-sealed base)
            b.instruction(specs.AND, X4, X2, Imm(SET_MASK))      # loaded value selects a probe set
            b.instruction(specs.LDR, X0, Mem(X4))                # DEPENDENT load (sandbox-clamped) -> set
            b.hole(Hole((1, 3), kind=Kind.ANY))                  # within if, after the probe (spec-only)

        b.hole(Hole((2, 6), kind=Kind.ALU))                      # AFTER the template (arch, non-mem)
