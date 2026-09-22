"""PTE environment leak: a spec-only read of the faulty page whose cache footprint depends on its PTE.

Store-bypass window (like canon_oracle_ssb): a SLOW store address (x4 = input x1, via a UDIV+MADD
chain) writes 0; a FAST load of the SAME location (base x1) bypasses the unresolved store and reads the
STALE (non-zero) value, so the B.ne is SPECULATIVELY taken while ARCHITECTURALLY not taken (the store
lands, flag 0). Inside that spec-only branch we read the faulty page (sandbox page 1) -- the branch
never retires, so faulty is reached ONLY speculatively.

With enable_pte_fuzzing, the genuine variant leaves faulty present (the spec read allocates a cache
line) while a decoy clears its `valid` bit (the spec read faults and allocates nothing). genuine and
decoy run byte-identical code and differ only in faulty's PTE, so the htrace divergence is a genuine
leak of page-table state through the speculative access.
"""
from src.aarch64.template import Template, Mem, Imm, specs, flags, X0, X1, X3, X4, X5, XZR

# Confine the architectural pointer to the main page, reserving the 8-byte access width: x1 in
# 0..0xFF8, so a 64-bit access ends by 0xFFF and cannot spill into faulty (offset 0x1000).
MAIN_PAGE_MASK = 0xFF8
FAULTY_PAGE_OFFSET = 0x1000   # sandbox page 0 = main; page 1 = faulty (the spec-only page)


class PteOracle(Template):
    def build(self, b):
        # Keep every RETIRING access in main; only the explicit x3 below reaches faulty, speculatively.
        b.instruction(specs.AND, X1, X1, Imm(MAIN_PAGE_MASK))
        b.instruction(specs.UDIV, X5, X1, X1)                    # x5 = 1
        for _ in range(15):
            b.instruction(specs.UDIV, X5, X5, X5)                # slow udiv chain (long SSB window)
        for _ in range(3):
            b.instruction(specs.MADD, X4, X1, X5, XZR)           # x4 = x1*x5 = x1, slow store address
        b.instruction(specs.STR, XZR, Mem(X4, offset=Imm(0)))    # SLOW store 0 -> arch flag 0
        b.instruction(specs.LDR, X0, Mem(X1, offset=Imm(0)))     # FAST load, same loc -> stale (!=0)
        b.instruction(specs.SUBS, XZR, X0, Imm(0))
        with b.if_(flags.NE):                                    # spec-taken (stale != 0), arch not
            b.instruction(specs.ORR, X3, XZR, Imm(FAULTY_PAGE_OFFSET))
            b.instruction(specs.LDR, X0, Mem(X3, offset=Imm(0)))  # spec-only read of the faulty page
