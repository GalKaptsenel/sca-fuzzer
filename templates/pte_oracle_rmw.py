"""PTE environment leak driven by a spec-only ATOMIC RMW of the faulty page (RMW Phase 5).

Same store-bypass window as pte_oracle, but the spec-only access to the faulty page is an LSE atomic
read-modify-write (LDADD) instead of a plain load. The RMW still READS the faulty page, so its cache
footprint depends on that page's PTE exactly as the plain load's does: with enable_pte_fuzzing the
genuine variant leaves faulty present (the speculative RMW allocates a line) while a decoy clears its
`valid` bit (the speculative access faults, allocating nothing). The branch never retires, so faulty is
reached only speculatively and the RMW's own store is squashed -- the divergence is a genuine leak of
page-table state through the speculative RMW.
"""
from src.aarch64.template import Template, Mem, Imm, specs, flags, X0, X1, X3, X4, X5, X6, XZR

MAIN_PAGE_MASK = 0xFF8         # confine the retiring pointer to main, reserving an 8-byte access
FAULTY_PAGE_OFFSET = 0x1000    # sandbox page 1 = faulty (the spec-only page)


class PteOracleRmw(Template):
    def build(self, b):
        b.instruction(specs.AND, X1, X1, Imm(MAIN_PAGE_MASK))
        b.instruction(specs.UDIV, X5, X1, X1)                    # x5 = 1
        for _ in range(15):
            b.instruction(specs.UDIV, X5, X5, X5)                # slow udiv chain (long SSB window)
        for _ in range(3):
            b.instruction(specs.MADD, X4, X1, X5, XZR)           # x4 = x1 (slow store address)
        b.instruction(specs.STR, XZR, Mem(X4, offset=Imm(0)))    # SLOW store 0 -> arch flag 0
        b.instruction(specs.LDR, X0, Mem(X1, offset=Imm(0)))     # FAST load, same loc -> stale (!=0)
        b.instruction(specs.SUBS, XZR, X0, Imm(0))
        with b.if_(flags.NE):                                    # spec-taken (stale != 0), arch not
            b.instruction(specs.ORR, X3, XZR, Imm(FAULTY_PAGE_OFFSET))
            # LDADD <Xs>, <Xt>, [<Xn>]: reads [X3] (faulty, spec-only) into X6, adds X5, stores back
            # (the store is squashed with the branch). The READ is the leaking access. Atomics take a
            # bare [Xn] with no displacement.
            b.instruction(specs.LDADD, X5, X6, Mem(X3))
