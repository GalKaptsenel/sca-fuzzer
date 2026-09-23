"""PTE environment leak driven by a spec-only CAS of the faulty page (RMW Phase 5, CAS variant).

Identical to pte_oracle_rmw but the spec-only faulty-page access is a compare-and-swap. CAS READS
[X3] (faulty) to compare against X5; that read is the leaking access whose cache footprint depends on
the faulty page's PTE. The conditional store is squashed with the never-retiring branch.
"""
from src.aarch64.template import Template, Mem, Imm, specs, flags, X0, X1, X3, X4, X5, X6, XZR

MAIN_PAGE_MASK = 0xFF8
FAULTY_PAGE_OFFSET = 0x1000


class PteOracleCas(Template):
    def build(self, b):
        b.instruction(specs.AND, X1, X1, Imm(MAIN_PAGE_MASK))
        b.instruction(specs.UDIV, X5, X1, X1)
        for _ in range(15):
            b.instruction(specs.UDIV, X5, X5, X5)
        for _ in range(3):
            b.instruction(specs.MADD, X4, X1, X5, XZR)
        b.instruction(specs.STR, XZR, Mem(X4, offset=Imm(0)))
        b.instruction(specs.LDR, X0, Mem(X1, offset=Imm(0)))
        b.instruction(specs.SUBS, XZR, X0, Imm(0))
        with b.if_(flags.NE):
            b.instruction(specs.ORR, X3, XZR, Imm(FAULTY_PAGE_OFFSET))
            b.instruction(specs.CAS, X5, X6, Mem(X3))    # compare X5 vs [X3] (spec-only read), swap X6
