"""TikTag-v1 (speculation-shrinkage) MTE template mirroring the official compsec-snu/tiktag Gadget 1,
shaped to be driven by the speculative-exception fuzzer with branch mistraining.

A slow input-dependent divide chain feeds a conditional branch so it resolves late (a long speculative
window) while branch mistraining forces the first execution to mispredict into the speculated block. In
that block a tag-checked load is repeated `LDNUM` times through an MTE-sealed base, then a serial
latency chain feeds a fixed probe load. On the genuine (matching-tag) variant the window stays open long
enough for the probe to issue and fill its L1 set; on a decoy (mismatching-tag) variant the tag check
shrinks the window so the probe never issues and the set stays absent. The tag-blind contract trace
keeps the probe present in both, so the genuine and decoy hardware traces diverge -> a speculative-exception
violation.

The window is a conditional-branch window (contract clause `cond`): the Cortex-X3 leaks v1 through
branch speculation shrinkage. The probe address is value-stable (the latency chain ORs in a zeroed
register), so it maps to one fixed cache set regardless of the input's memory contents, and its
divergence is a binary present/absent rather than a data-dependent set selection.
"""
from src.aarch64.template import Template, Hole, Kind, Mem, Imm, specs, flags, \
    X0, X1, X2, X3, X4, X5, X6, XZR

PROBE_OFF = 2048        # fixed probe address -> cache set 32 (2048 / 64)
LDNUM = 8               # repeated tag-checked loads: more speculative checks deepen the window shrink
GAP = 64               # serial latency-chain length pacing the probe against the window shrink


class MteTiktagG1(Template):
    def build(self, b):
        b.hole(Hole((2, 6), kind=Kind.ALU))                   # BEFORE (architectural, non-memory)

        b.instruction(specs.SDIV, X0, X1, X2)                 # slow input-dependent divide (window opener)
        for _ in range(7):
            b.instruction(specs.SDIV, X0, X0, X2)             # dependent divide chain: branch resolves late
        b.instruction(specs.SUBS, XZR, X0, Imm(0))
        with b.if_(flags.NE):                                 # mistrained to mispredict -> speculated block
            for _ in range(LDNUM):
                b.instruction(specs.LDR, X2, Mem(X3))         # tag-checked guess load (X3 MTE-sealed)
            b.instruction(specs.ORR, X6, XZR, XZR)            # X6 = 0 in a register (not XZR)
            b.instruction(specs.ORR, X4, XZR, Imm(PROBE_OFF))  # X4 = fixed probe offset
            for _ in range(GAP):
                b.instruction(specs.ORR, X4, X4, X6)          # serial latency chain (value-stable, X6 == 0)
            b.instruction(specs.LDR, X5, Mem(X4))             # probe load -> fixed set, present iff reached

        b.hole(Hole((2, 6), kind=Kind.ALU))                   # AFTER (architectural, non-memory)
