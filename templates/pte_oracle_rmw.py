"""PTE environment leak driven by a spec-only ATOMIC RMW, with random holes around the gadget.

Store-bypass window (Spectre-v4/SSB): a slow store's address stays unresolved while a fast load of the
same location bypasses it and reads the STALE (non-zero) value, so the branch is SPECULATIVELY taken
while ARCHITECTURALLY not taken -- the branch never retires. Inside that spec-only branch an LSE atomic
RMW (LDADD) accesses memory through a REGULARLY-SANDBOXED, input-derived base: the standard sandbox
clamp spans main+faulty, so the address lands wherever the input puts it -- MAYBE in the faulty page.
The NI/PTE fuzzing is what turns a faulty landing into a leak: genuine keeps the faulty page present
(the spec RMW allocates a line) while a decoy flips a leaf PTE bit (the spec access faults, allocating
nothing). Because the access is speculative-only it never retires, so a faulting decoy page is safe.

The architectural path is confined to main (x1 masked to 0xFF8), so it never touches the faulty page --
only the speculative RMW can, which is what the arch-safety guard requires.

Registers: only the 6 whitelisted, input-seeded registers x0-x5. Holes: before/after the template are
ARCHITECTURAL so Kind.ALU only (an arch memory access could hit the faulty page and, under a decoy PTE,
fault-and-panic / trip the arch-safety guard); the holes inside the if are SPECULATIVE-only so Kind.ANY.
The gadget re-establishes every register it uses, so a hole clobbering x0-x5 cannot break the leak.
"""
from src.aarch64.template import Template, Hole, Kind, Mem, Imm, specs, flags, X0, X1, X2, X3, X4, X5, XZR

MAIN_PAGE_MASK = 0xFF8         # confine the retiring pointer to main, reserving an 8-byte access


class PteOracleRmw(Template):
    def build(self, b):
        b.hole(Hole((2, 6), kind=Kind.ALU))                      # BEFORE the template (arch, non-mem)

        b.instruction(specs.AND, X1, X1, Imm(MAIN_PAGE_MASK))    # confine x1 (the arch path) to main
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
            # Regularly-sandboxed, input-derived spec-only RMW: base x3 is an input register; the standard
            # sandbox clamp (main+faulty) decides where it lands, so it MAY hit the faulty page per input.
            b.instruction(specs.LDADD, X5, X2, Mem(X3))          # spec-only atomic RMW; dest x2 (within x0-x5)
            b.hole(Hole((1, 4), kind=Kind.ANY))                  # within if, AFTER the access (spec-only)

        b.hole(Hole((2, 6), kind=Kind.ALU))                      # AFTER the template (arch, non-mem)
