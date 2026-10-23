"""Task-1 controls for the TikTag-v1 executor detector (Fable verification).

Four independent soundness checks, all through /dev/executor on the X3 (core 8), run strictly serially:

  (A) reach/skip/spec: prove the F+R htrace resolves "probe reached" (~1.0) vs "not reached" (~0.0)
      and that a PURELY SPECULATIVE probe (real gadget, matching tag) registers.
  (B) read-set independence: rerun the match-vs-mismatch sweep at a DIFFERENT read set (24) to show
      the surviving tag is still the sealed tag and the signal appears at set 24, not fixed at set 32.
  (C) graded refills: dump /sys/executor/reload_refills for match vs checked-mismatch to see per-set
      L1 (counter0) and L2 (counter1) refill deltas -- is a mismatch a full DRAM miss or just L2?
  (D) finer clock: rerun the match-vs-mismatch sweep with reload_timer_shift set to a cycle threshold
      to check whether a graded cycle readout changes the verdict.
"""
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
DEV = "/data/local/tmp"
EU = f"{DEV}/executor_userland /dev/executor"
T, LDNUM, GAP, ITERS, N = 5, 16, 80, 128, 40
LOG = open(f"{OUT}/g1_control_fable_results.txt", "w")


def out(*a):
    s = " ".join(str(x) for x in a); print(s); LOG.write(s + "\n"); LOG.flush()


d = Client('127.0.0.1', 5037).devices()[0]


def su(c):
    return d.shell(f"su -c '{c}'")


MAIN = int(su("cat /sys/executor/print_sandbox_base").strip(), 16)


def compile_asm(asm, tag):
    ap = f"{OUT}/ctl_{tag}.asm"; bp = f"{OUT}/ctl_{tag}.bin"
    open(ap, "w").write(asm)
    r = subprocess.run([PY, COMPILE, "--input", ap, "--output", bp], capture_output=True)
    if r.returncode:
        out("COMPILE FAIL", tag, r.stderr.decode()); sys.exit(1)
    return bp


def loop_gadget(read_set, warm_set):
    L = [f"mov x3, #{ITERS}", ".Lloop:", "subs x3, x3, #1",
         "csel x6, x5, x1, eq", "csel x10, x2, x4, eq", "cset x7, ne",
         "ldr x12, [x1]", "str x7, [x0]",
         "dc civac, x0", "dc civac, x2", "dc civac, x4", "dsb sy", "isb",
         "ldr x8, [x0]", "cbz x8, .Lskip"]
    L += ["ldr x9, [x6]"] * LDNUM + ["orr x10, x10, x9"] * GAP
    L += ["ldr x11, [x10]", ".Lskip:", "cbnz x3, .Lloop"]
    return "\n".join(L) + "\n"


REACH = "dc civac, x2\ndsb sy\nisb\nldr x11, [x2]\n"
SKIP = "dc civac, x2\ndsb sy\nisb\nnop\n"


def mkinp(read_set, warm_set, G):
    inp = Input(1); inp.view('uint64')[:] = 0
    g = inp[0]['gpr']
    g[0] = MAIN + 0x400
    g[1] = (MAIN & ~(0xF << 56)) | (T << 56)
    g[2] = MAIN + read_set * 64
    g[4] = MAIN + warm_set * 64
    g[5] = (MAIN & ~(0xF << 56)) | (G << 56)
    return inp


def tags():
    t = [15] * 512; t[0] = T; return t


def regime(timer_shift=-1):
    for k, v in (("measurement_mode", "F+R"), ("pin_to_core", "8"), ("warmups", "1"),
                 ("enable_pre_run_flush", "1"), ("enable_phr_flush", "0"),
                 ("enable_view_rotation", "0"), ("enable_ssbs", "0"),
                 ("reload_timer_shift", str(timer_shift))):
        su(f"echo {v} > /sys/executor/{k}")


def load_tc(binf):
    d.push(binf, binf.replace(OUT, DEV)); su(f"chmod 666 {binf.replace(OUT, DEV)}")
    su(f"{EU} 1 >/dev/null; {EU} w {binf.replace(OUT, DEV)} >/dev/null")


def push_reif(name, inp):
    ei = ExecutorInput(input_=inp, mte_tags=tags())
    p = f"{OUT}/{name}.reif"; open(p, "wb").write(ei.serialize())
    d.push(p, f"{DEV}/{name}.reif"); su(f"chmod 666 {DEV}/{name}.reif")
    return f"{DEV}/{name}.reif"


def residency(reif, read_set, reps=N):
    o = su(f"sh {DEV}/run_tag.sh {reif} {reps}")
    hs = re.findall(r"htrace 0:\s*([01]{64})", o)
    idx = 63 - read_set
    return (sum(int(h[idx]) for h in hs) / len(hs)) if hs else -1.0


d.push(f"{OUT}/run_tag.sh", f"{DEV}/run_tag.sh"); su(f"chmod 755 {DEV}/run_tag.sh")
out(f"MAIN={hex(MAIN)} T={T} ldnum={LDNUM} gap={GAP} iters={ITERS} reps={N} core=8")

# ---- (A) reach / skip / spec ----
regime(-1)
out("\n=== (A) reach / skip / spec (read set 32) ===")
load_tc(compile_asm(REACH, "reach"))
r = push_reif("ctl_reach", mkinp(32, 48, T))
out(f"  reach  (architectural probe to set 32): residency {residency(r, 32):.2f}  [expect ~1.0]")
load_tc(compile_asm(SKIP, "skip"))
r = push_reif("ctl_skip", mkinp(32, 48, T))
out(f"  skip   (no probe):                      residency {residency(r, 32):.2f}  [expect ~0.0]")
load_tc(compile_asm(loop_gadget(32, 48), "spec"))
rm = push_reif("ctl_spec_m", mkinp(32, 48, T))
rx = push_reif("ctl_spec_x", mkinp(32, 48, 0))
out(f"  spec   (match tag {T}, probe speculative-only): residency {residency(rm, 32):.2f}  [expect high]")
out(f"  spec   (mismatch tag 0):                        residency {residency(rx, 32):.2f}  [expect ~0.0]")

# ---- (B) read-set independence (read set 24) ----
out("\n=== (B) read-set independence: same sweep at read set 24 ===")
load_tc(compile_asm(loop_gadget(24, 48), "rs24"))
res = {}
for G in (0, 3, T, 9, 15):
    rr = push_reif(f"ctl_rs24_{G}", mkinp(24, 48, G))
    res[G] = residency(rr, 24)
out("  " + " ".join(f"{g}:{res[g]:.2f}" for g in (0, 3, T, 9, 15)) + f"  (sealed T={T})")
mm = [res[g] for g in (0, 3, 9) ]
out(f"  match {T}={res[T]:.2f}  checked-mm max={max(mm):.2f}  tcma15={res[15]:.2f}  "
    f"sep={res[T]-max(mm):+.2f}")

# ---- (C) graded refills (match vs mismatch, read set 32) ----
out("\n=== (C) graded refills: per-set L1/L2 refill deltas (read set 32) ===")
load_tc(compile_asm(loop_gadget(32, 48), "ref"))
for label, G in (("match T=5", T), ("mismatch 0", 0), ("tcma 15", 15)):
    reif = push_reif(f"ctl_ref_{G}", mkinp(32, 48, G))
    # one TRACE + one MEASUREMENT so reload_refills reflects a single reload of this input
    iid = su(f"{EU} 5").strip()
    m = re.search(r'Allocated Input ID[^0-9]*([0-9]+)', iid)
    iid = m.group(1) if m else "0"
    su(f"{EU} 4 {iid} >/dev/null; {EU} w {reif} >/dev/null")
    su(f"{EU} 8 >/dev/null; {EU} 4 {iid} >/dev/null; {EU} 7 >/dev/null")
    line = su("cat /sys/executor/reload_refills | sed -n '33p'")  # set 32 (0-indexed line 33)
    out(f"  {label:12s} set32  [set L1main L1faulty L2main L2faulty] = {line.strip()}")
su(f"{EU} 9 >/dev/null 2>&1")

# ---- (D) finer clock: cycle-threshold readout ----
out("\n=== (D) finer clock: match vs mismatch at reload_timer_shift cycle thresholds ===")
load_tc(compile_asm(loop_gadget(32, 48), "clk"))
for shift in (7, 8, 9):
    regime(shift)
    rm = push_reif("ctl_clk_m", mkinp(32, 48, T))
    rx = push_reif("ctl_clk_x", mkinp(32, 48, 0))
    rt = push_reif("ctl_clk_t", mkinp(32, 48, 15))
    out(f"  shift={shift} (<{1<<shift} cyc=hit): match={residency(rm,32):.2f} "
        f"mismatch0={residency(rx,32):.2f} tcma15={residency(rt,32):.2f}")
regime(-1)
out("\nDONE")
