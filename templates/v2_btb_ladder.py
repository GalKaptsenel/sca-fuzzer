"""Spectre-v2 (indirect branch) over the BTB -- DE-STRUCTURING LADDER.

Starts from templates/spectre_v2_btb.py (the strictly-cross-input, pure-which-target BTB gadget that random
fuzzing finds only rarely) and removes structure one level at a time toward fully-random code, to find WHICH
ingredient is the bottleneck for auto-detecting the strict cross-input pure-BTB leak under v2_btb_seq.yml
(CT-SEQ + cross_input_leaks_only). Set LEVEL and run; each level ADDS randomization on top of the previous.

  L0  exact template: 3 fixed targets return 512/1024/2048; 15x UDIV slow chain; STR(slow)/LDR(stale) SSB
      window; if_(NE) arch-not-taken guard; BLR dispatch on stale x0; readout LDR from returned x3.
  L1  + random distinct return offsets (still distinct L1D sets, still no memory in targets).
  L2  + random target count (2..4) and random NON-MEMORY (Kind.ALU) bodies in each target (still pure BTB).
  L3  + random noise holes (Kind.ANY) before and after the gadget in the entry.
  L4  + random-LENGTH UDIV slow chain (5..20) instead of a fixed 15.
  L5  + random-KIND delay: an ALU hole on x5 instead of the UDIV chain (tests whether long store-address
      latency is essential to opening the SSB window).
  L6  + random guard condition: if_() with no pinned condition (a random conditional branch) instead of
      if_(flags.NE).

The essential, irremovable ingredients (hypothesis): the SSB-gated spec-only dispatch whose index is
UNTAINTED under CT-SEQ (so boosting varies it across lanes = the cross-input which-target toggle). The
level at which detection breaks pins the bottleneck.

RESULT (N3, eval_configs/v2_btb_seq.yml, 2026-10-17): NOTHING fully breaks -- every level still auto-detects
the strict cross-input leak -- but reliability degrades MONOTONICALLY as structure is removed:
  L0..L4  confirm in ~1 test case (tens of seconds): offsets, target count, non-memory target bodies,
          surrounding noise, and slow-chain LENGTH are all cosmetic.
  L5      confirms but needs several test cases (~4-5 min): a random low-latency ALU delay opens the SSB
          window less reliably than the long UDIV latency chain.
  L6      confirms only after ~512 s: a random guard DECOUPLED from the stale load relies on N3 v1 branch
          misprediction (unreliable) instead of the SSB window forcing the branch spec-taken.
So the structure is not binary-essential; it TUNES reliability. The two reliability levers are (1) long
store-ADDRESS latency and (2) the SSB-DRIVEN guard (condition computed from the stale load). Fully-random
code hits that combination rarely -- which is exactly why fully-random strict-cross-input pure-BTB is rare,
while plain v2 (self-dependent) is abundant (see [[project_v2_btb_random_filter_bug]]).
"""
import random

from src.aarch64.template import (Template, Hole, Kind, IndirectCall, Mem, Imm, specs, flags,
                                  X0, X1, X3, X4, X5, XZR)

LEVEL = 0


def _target(off, level):
    def body(f):
        if level >= 2:
            f.hole(Hole(n=(1, 3), kind=Kind.ALU))          # random non-memory work (stays pure BTB)
        f.instruction(specs.ORR, X3, XZR, Imm(off))        # return a distinct single-bit offset in x3
    return body


class SpectreV2BtbLadder(Template):
    def build(self, b):
        level = LEVEL

        n_tgt = random.randint(2, 4) if level >= 2 else 3
        if level >= 1:
            # distinct power-of-two offsets (valid ORR bitmask immediates) -> distinct L1D sets 2/4/8/16/32
            offs = random.sample([128, 256, 512, 1024, 2048], n_tgt)
        else:
            offs = [512, 1024, 2048]
        funcs = [b.function(_target(o, level)) for o in offs]

        if level >= 3:
            b.hole(Hole(n=(2, 6), kind=Kind.ANY))          # noise before the gadget

        b.instruction(specs.UDIV, X5, X1, X1)              # x5 = 1
        if level >= 5:
            b.hole(Hole(n=(8, 20), kind=Kind.ALU, regs=[X5]))   # random low-latency delay (no long UDIV)
        else:
            reps = random.randint(5, 20) if level >= 4 else 15
            for _ in range(reps):
                b.instruction(specs.UDIV, X5, X5, X5)      # slow 12-cycle dependent chain
        for _ in range(3):
            b.instruction(specs.MADD, X4, X1, X5, XZR)     # x4 = x1*x5 = x1, store address (slow to resolve)
        b.instruction(specs.STR, XZR, Mem(X4, offset=Imm(0)))   # SLOW store 0 -> arch guard 0
        b.instruction(specs.LDR, X0, Mem(X1, offset=Imm(0)))    # FAST load same loc -> stale (!=0) spec
        b.instruction(specs.SUBS, XZR, X0, Imm(0))

        guard = None if level >= 6 else flags.NE           # L6: random conditional instead of pinned NE
        with b.if_(guard):
            b.call(IndirectCall(targets=funcs, dispatch=True, index_reg=X0))
            b.instruction(specs.LDR, X0, Mem(X3, offset=Imm(0)))   # readout: which-target -> L1D set

        if level >= 3:
            b.hole(Hole(n=(2, 6), kind=Kind.ANY))          # noise after the gadget
