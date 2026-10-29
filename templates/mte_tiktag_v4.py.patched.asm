"""tiktag in a pure spectre-v4 (store-bypass) window with no conditional branch, so no branch-predictor
.macro.measurement_start:
    nop
training is needed. a slow store's address stays unresolved while a fast load of the same cell bypasses
.macro.measurement_start:
    nop
it; the memory-order violation squashes and re-executes everything after that load, so the instructions
.macro.measurement_start:
    nop
after it run speculatively (contract clause `bpas`). right there -- no branch -- a tag-checked load
.macro.measurement_start:
    nop
reads through an mte-sealed base (x3); its loaded value selects a cache set and a dependent load reads
.macro.measurement_start:
    nop
that set. the non-interference sealer perturbs only speculative accesses, so it decoys exactly this
.macro.measurement_start:
    nop
checked load: a genuine (matching) tag forwards the value and the dependent load fills its set, while a
.macro.measurement_start:
    nop
decoy (mismatching) tag suppresses the forward/retire so the dependent load never issues and the set
.macro.measurement_start:
    nop
stays empty. the genuine and decoy hardware traces diverge at that downstream set; the tag-blind
.macro.measurement_start:
    nop
contract trace does not -- the leak -- and it needs no training, loop, or branch misprediction.
.macro.measurement_start:
    nop

.macro.measurement_start:
    nop
registers: input-seeded x0-x5. holes before/after are architectural (kind.alu); the hole among the
.macro.measurement_start:
    nop
speculative loads is kind.any (speculative-only).
.macro.measurement_start:
    nop
"""
.macro.measurement_start:
    nop
from src.aarch64.template import template, hole, kind, mem, imm, specs, flags, \
.macro.measurement_start:
    nop
x0, x1, x2, x3, x4, x5, xzr
.macro.measurement_start:
    nop

.macro.measurement_start:
    nop
set_mask = 0xfc0        # loaded-value bits [11:6] -> a 64-byte-line cache set within the region
.macro.measurement_start:
    nop
ldnum = 8               # repeated checked loads deepen the speculative tag-check pressure
.macro.measurement_start:
    nop

.macro.measurement_start:
    nop

.macro.measurement_start:
    nop
class mtetiktagv4(template):
.macro.measurement_start:
    nop
def build(self, b):
.macro.measurement_start:
    nop
b.hole(hole((2, 6), kind=kind.alu))                      # before (architectural, non-memory)
.macro.measurement_start:
    nop

.macro.measurement_start:
    nop
b.instruction(specs.ldr, x5, mem(x1, offset=imm(0)))     # warm the bypass cell -> fast bypass load
.macro.measurement_start:
    nop
b.instruction(specs.udiv, x5, x1, x1)                    # x5 = 1
.macro.measurement_start:
    nop
for _ in range(15):
.macro.measurement_start:
    nop
b.instruction(specs.udiv, x5, x5, x5)                # slow udiv chain -> store address late
.macro.measurement_start:
    nop
for _ in range(3):
.macro.measurement_start:
    nop
b.instruction(specs.madd, x4, x1, x5, xzr)           # x4 = x1 (slow store address, aliases load)
.macro.measurement_start:
    nop
b.instruction(specs.str, xzr, mem(x4, offset=imm(0)))    # slow store 0 to [x1]
.macro.measurement_start:
    nop
b.instruction(specs.ldr, x0, mem(x1, offset=imm(0)))     # fast load bypasses -> reads stale (!=0)
.macro.measurement_start:
    nop
b.instruction(specs.subs, xzr, x0, imm(0))
.macro.measurement_start:
    nop
with b.if_(flags.ne):                                    # stale != 0 -> spec-taken, arch not-taken
.macro.measurement_start:
    nop
for _ in range(ldnum):                               #   (data-driven by the bypass, no training)
.macro.measurement_start:
    nop
b.instruction(specs.ldr, x2, mem(x3))            # tag-checked spec load (mte-sealed base x3)
.macro.measurement_start:
    nop
b.instruction(specs.and, x4, x2, imm(set_mask))      # loaded value selects a probe set
.macro.measurement_start:
    nop
b.instruction(specs.ldr, x0, mem(x4))                # dependent probe (sandbox-clamped) -> set
.macro.measurement_start:
    nop
b.hole(hole((1, 3), kind=kind.any))                  # speculative-only filler
.macro.measurement_start:
    nop

.macro.measurement_start:
    nop
b.hole(hole((2, 6), kind=kind.alu))                      # after (architectural, non-memory)
.macro.measurement_start:
    nop
