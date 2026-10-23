"""SSB-window TikTag test: create the speculative window with a store-bypass (Spectre-v4), NOT a
trained branch, so no branch-direction training is needed (sidesteps the batch-pollution problem).
A slow store (address through a UDIV chain) is bypassed by a load of the same cell, whose STALE value
makes a branch fall through into a speculative block holding ldnum tag-checked sealed loads, a serial
orr-chain latency gate, and a fixed probe. If the MTE tag-check shrink acts in an SSB window, the
matching tag keeps the probe present while mismatches are gated. Self-inflicted single-input tag sweep."""
import sys, re, subprocess, os
REPO = "/home/gal_k_1_1998/revizor/sca-fuzzer"
sys.path.insert(0, REPO)
from src.config import CONF
CONF.load(os.path.join(REPO, "config_g1_x3.yml"))
from src.interfaces import Input
from src.aarch64.aarch64_executor_input_encoder import ExecutorInput
from ppadb.client import Client

OUT = "/tmp/claude-1000/-home-gal-k-1-1998-revizor-sca-fuzzer/cf5e0fed-2c4e-46af-8f5e-49fcf726976a/scratchpad"
COMPILE = os.path.join(REPO, "src/executor_userland/asm_compiler/compile_test_case.py")
PY = "/home/gal_k_1_1998/revizor/revizor-venv/bin/python"
T, READ_SET, LDNUM, N = 5, 32, 16, 40
GAPS = [20, 40, 80]
DEV = "/data/local/tmp"
EU = f"{DEV}/executor_userland /dev/executor"
LOG = open(f"{OUT}/g1_ssb_results.txt", "w")


def out(*a):
    s = " ".join(str(x) for x in a); print(s); LOG.write(s + "\n"); LOG.flush()


d = Client('127.0.0.1', 5037).devices()[0]


def su(c):
    return d.shell(f"su -c '{c}'")


MAIN = int(su("cat /sys/executor/print_sandbox_base").strip(), 16)


def gadget_asm(gap):
    L = ["ldr x12, [x4]",              # warm guess granule (unchecked alias)
         "ldr x13, [x0]",              # WARM the SSB cell so the bypassing load is FAST (L1 hit)
         "dc civac, x2",               # flush probe read set only (NOT x0)
         "udiv x5, x0, x0"]            # x5 = 1 (x0 is a pointer)
    L += ["udiv x5, x5, x5"] * 15      # slow dependent chain -> delays the store address
    L += ["madd x7, x0, x5, xzr",      # store addr = x0 * 1 = x0 (aliases the load), resolves late
          "str xzr, [x7]",             # SLOW store (writes 0)
          "ldr x8, [x0]",              # fast load bypasses the store -> STALE (non-zero) value
          "subs xzr, x8, #0",
          "b.eq .Lskip"]               # stale != 0 -> NE -> fall through (spec); resolved 0 -> skip
    L += ["ldr x9, [x1]"] * LDNUM      # tag-checked sealed loads (x1 = guess ptr, swept tag)
    L += ["orr x6, xzr, xzr"] + ["orr x2, x2, x6"] * gap   # serial latency gate on probe addr
    L += ["ldr x11, [x2]", ".Lskip:"]
    return "\n".join(L) + "\n"


def build_tc(gap):
    asm = f"{OUT}/g1_ssb{gap}.asm"; binf = f"{OUT}/g1_ssb{gap}.bin"
    open(asm, "w").write(gadget_asm(gap))
    r = subprocess.run([PY, COMPILE, "--input", asm, "--output", binf], capture_output=True)
    if r.returncode:
        out(r.stderr.decode()); sys.exit(1)
    return binf


def mkinp(G):
    inp = Input(1); inp.view('uint64')[:] = 0
    g = inp[0]['gpr']
    g[0] = MAIN + 0x600                          # SSB alias cell (set 24); its granule is unchecked
    inp[0]['main'][0x600 // 8] = 1               # stale value = 1 (non-zero -> NE -> enter spec block)
    g[1] = (MAIN & ~(0xF << 56)) | (G << 56)     # guess ptr, swept tag (set 0)
    g[2] = MAIN + READ_SET * 64                  # probe (set 32)
    g[4] = (MAIN & ~(0xF << 56)) | (15 << 56)    # unchecked warm alias to the guess granule
    t = [15] * 512; t[0] = T                     # seal guess granule 0 to tag T
    return ExecutorInput(input_=inp, mte_tags=t).serialize()


d.push(f"{OUT}/run_tag.sh", f"{DEV}/run_tag.sh"); su(f"chmod 755 {DEV}/run_tag.sh")
for G in range(16):
    open(f"{OUT}/g1_in{G}.reif", "wb").write(mkinp(G))
    d.push(f"{OUT}/g1_in{G}.reif", f"{DEV}/g1_in{G}.reif")
su(f"chmod 666 {DEV}/g1_in*.reif")
# SSB enabled (store bypass happens), pinned X3, no flush knobs
for k, v in (("measurement_mode", "F+R"), ("pin_to_core", "8"), ("warmups", "1"),
             ("enable_pre_run_flush", "1"), ("enable_phr_flush", "0"),
             ("enable_view_rotation", "0"), ("enable_ssbs", "1")):
    su(f"echo {v} > /sys/executor/{k}")

idx = 63 - READ_SET
out(f"MAIN={hex(MAIN)} T={T} read={READ_SET} ldnum={LDNUM} reps={N} core=8 SSB=on (store-bypass window)")
for gap in GAPS:
    binf = build_tc(gap)
    d.push(binf, binf.replace(OUT, DEV)); su(f"chmod 666 {binf.replace(OUT, DEV)}")
    su(f"{EU} 1 >/dev/null; {EU} w {binf.replace(OUT, DEV)} >/dev/null")
    res = {}
    for G in range(16):
        o = su(f"sh {DEV}/run_tag.sh {DEV}/g1_in{G}.reif {N}")
        hs = re.findall(r"htrace 0:\s*([01]{64})", o)
        res[G] = (sum(int(h[idx]) for h in hs) / len(hs)) if hs else -1.0
    mm = [res[g] for g in range(16) if g != T and g != 15]
    sep = res[T] - max(mm)
    out(f"--- gap={gap} ---")
    out(" ".join(f"{g}:{res[g]:.2f}" for g in range(16)))
    out(f"   MATCH={res[T]:.2f} checked-mm max={max(mm):.2f} mean={sum(mm)/len(mm):.2f} sep={sep:+.2f}"
        f" {'*** SSB window DETECTS (no training) ***' if sep > 0.15 else 'no separation'}")
