"""MTE tag-check leak driven by a spec-only load behind a store-bypass window, with random holes
around the gadget.

Store-bypass window (Spectre-v4/SSB): a slow store's address stays unresolved while a fast load of the
same location bypasses it and reads the STALE (non-zero) value, so the branch is SPECULATIVELY taken
while ARCHITECTURALLY not taken -- the branch never retires. Inside that spec-only branch a load reads
through a regularly-sandboxed, input-derived base. The MTE seal retags that pointer onto a granule tag
for the one access: genuine matches the accessed cell's allocation tag, a decoy mismatches it. The
speculative tag check gates whether the load's line allocates, so a genuine match and a decoy mismatch
diverge in the cache htrace while the tag-blind contract trace does not -- the leak. The tag match/
mismatch is the fuzzed precondition.

The seal lands only on speculative-only slots, so the architectural store and fast load keep the cell's
real tag and never fault. The speculative load never retires, so a mismatched decoy tag is safe.

Registers: only the input-seeded x0-x5. Holes before and after the template are architectural, so
Kind.ALU only; the holes inside the if are speculative-only, so Kind.ANY. The gadget re-establishes
every register it uses, so a hole clobbering x0-x5 cannot break the leak.
"""
from src.aarch64.template import Template, Hole, Kind, Mem, Imm, specs, flags, X0, X1, X2, X3, X4, X5, XZR


class MteOracle(Template):
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
            b.hole(Hole((1, 4), kind=Kind.ANY))                  # within if, BEFORE the access (spec-only)
            b.instruction(specs.LDR, X2, Mem(X3))                # spec-only load; MTE-sealed base x3
            b.hole(Hole((1, 4), kind=Kind.ANY))                  # within if, AFTER the access (spec-only)

        b.hole(Hole((2, 6), kind=Kind.ALU))                      # AFTER the template (arch, non-mem)
