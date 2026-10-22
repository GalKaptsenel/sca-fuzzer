#!/usr/bin/env python3
"""Generate a loop-free DAG TikTag-v1 NI template faithful to the manual PoC: an architectural
warm+zero of the guess granule (fixed probe set 32, warm L1 hit), a deep SDIV compute-delay window
feeding an arch-taken forward branch (mistrained not-taken -> speculative body), `ldnum` tag-checked
guess loads, a `gap`-long data-dependent ORR gate chaining the probe address through the guess result,
then the probe. The body sits on the branch's arch-not-taken path so its loads are sealer-eligible
(speculative-only) and the MTE sealer decoys them with a mismatched tag.

Usage: gen_s2.py --ldnum 16 --gap 80 --sdiv 24 --out <path.asm> [--no-warm]
"""
import argparse


def gen(ldnum, gap, sdiv, warm, det_window=False):
    L = [".test_case_enter:", ".section .data.main", ".function_0:", ".bb_0.0:",
         ".macro.measurement_start: NOP"]
    if warm:
        L += ["STR xzr, [x5]",            # zero the guess granule (arch, matching tag)
              "LDR x12, [x5]"]            # warm it to L1 (arch) -> x12 = 0
    L += ["ORR x10, xzr, #2048"]         # probe base -> set 32 (0x800/64)
    if det_window:
        # input-INDEPENDENT window: fixed divisor so every input/rep gets the same SDIV latency chain
        L += ["ORR x6, xzr, #7", "ORR x9, xzr, #65535", "SDIV x9, x9, x6"]
        L += ["SDIV x9, x9, x6"] * (sdiv - 1)
    else:
        L += ["SDIV x9, x1, x2"]
        L += ["SDIV x9, x9, x2"] * (sdiv - 1)
    L += ["ORR x9, x9, #1",              # force arch nonzero -> branch arch TAKEN
          "CBNZ x9, .bb_0.2",            # arch taken -> skip body; mistrained not-taken -> spec body
          ".bb_0.1:"]
    L += ["LDR x12, [x5]"] * ldnum       # tag-checked guess loads (warm L1 hit, value 0)
    L += ["AND x11, x12, xzr"]           # x11 = 0 but DATA-DEPENDENT on the guess result x12
    L += ["ORR x10, x10, x11"] * gap     # serial latency gate: probe waits on the guess, addr stays set 32
    L += ["LDR x13, [x10]",              # probe -> FIXED set 32 (0x800), present iff the window reaches it
          ".bb_0.2:",
          ".macro.measurement_end: NOP",
          "B .test_case_exit",
          ".section .data.main",
          ".test_case_exit:"]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ldnum", type=int, default=16)
    ap.add_argument("--gap", type=int, default=80)
    ap.add_argument("--sdiv", type=int, default=24)
    ap.add_argument("--no-warm", action="store_true")
    ap.add_argument("--det-window", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    with open(a.out, "w") as f:
        f.write(gen(a.ldnum, a.gap, a.sdiv, not a.no_warm, a.det_window))
    print(f"wrote {a.out} (ldnum={a.ldnum} gap={a.gap} sdiv={a.sdiv} warm={not a.no_warm} det={a.det_window})")


if __name__ == "__main__":
    main()
