"""PTE environment leak driven by a spec-only CASP (128-bit pair) of the faulty page (RMW Phase 5).

Identical to pte_oracle_rmw but the spec-only faulty-page access is a compare-and-swap PAIR: it READS
the 16 bytes at [X3]..[X3]+8 (faulty), the leaking access. The generator's _patch_casp fixes the four
data operands to consecutive even/odd pairs; CASP's 16-byte access is 16-byte aligned (X3 = 0x1000).
"""
from src.aarch64.template import Template, Mem, Imm, specs, flags, X0, X1, X2, X3, X4, X5, X6, X7, XZR

MAIN_PAGE_MASK = 0xFF8
FAULTY_PAGE_OFFSET = 0x1000


class PteOracleCasp(Template):
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
            # CASP Rs,Rs+1,Rt,Rt+1,[Xn]; _patch_casp rewrites to consecutive even/odd pairs off-base.
            b.instruction(specs.CASP, X0, X1, X6, X7, Mem(X3))
