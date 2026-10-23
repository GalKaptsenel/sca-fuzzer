"""Diagnostic: why does the tight internal-loop gadget detect while the cross-input batch does not?
Sweep the internal training-loop iteration count (gap=80, ldnum=16, sealed tag 5) and measure the
match-vs-mismatch separation. If detection survives at ~64 iters, the batch's failure at K=64 is
pollution (harness branches between inputs); if it needs ~256+, the batch simply provides too few
(effective) trainings. Uses the fast on-device run_tag.sh."""
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
T, READ_SET, WARM_SET, LDNUM, GAP, N = 5, 32, 48, 16, 80, 40
ITERS_LIST = [8, 32, 64, 128, 256, 512]
DEV = "/data/local/tmp"
EU = f"{DEV}/executor_userland /dev/executor"
LOG = open(f"{OUT}/g1_iters_results.txt", "w")


def out(*a):
    s = " ".join(str(x) for x in a); print(s); LOG.write(s + "\n"); LOG.flush()


d = Client('127.0.0.1', 5037).devices()[0]


def su(c):
    return d.shell(f"su -c '{c}'")


MAIN = int(su("cat /sys/executor/print_sandbox_base").strip(), 16)


def gadget_asm(iters):
    L = [f"mov x3, #{iters}", ".Lloop:", "subs x3, x3, #1",
         "csel x6, x5, x1, eq", "csel x10, x2, x4, eq", "cset x7, ne",
         "ldr x12, [x1]", "str x7, [x0]",
         "dc civac, x0", "dc civac, x2", "dc civac, x4", "dsb sy", "isb",
         "ldr x8, [x0]", "cbz x8, .Lskip"]
    L += ["ldr x9, [x6]"] * LDNUM + ["orr x10, x10, x9"] * GAP
    L += ["ldr x11, [x10]", ".Lskip:", "cbnz x3, .Lloop"]
    return "\n".join(L) + "\n"


def build_tc(iters):
    asm = f"{OUT}/g1_it{iters}.asm"; binf = f"{OUT}/g1_it{iters}.bin"
    open(asm, "w").write(gadget_asm(iters))
    r = subprocess.run([PY, COMPILE, "--input", asm, "--output", binf], capture_output=True)
    if r.returncode:
        out(r.stderr.decode()); sys.exit(1)
    return binf


def mkinp(G):
    inp = Input(1); inp.view('uint64')[:] = 0
    g = inp[0]['gpr']
    g[0] = MAIN + 0x400
    g[1] = (MAIN & ~(0xF << 56)) | (T << 56)
    g[2] = MAIN + READ_SET * 64
    g[4] = MAIN + WARM_SET * 64
    g[5] = (MAIN & ~(0xF << 56)) | (G << 56)
    return inp


def tags():
    t = [15] * 512; t[0] = T; return t


d.push(f"{OUT}/run_tag.sh", f"{DEV}/run_tag.sh"); su(f"chmod 755 {DEV}/run_tag.sh")
for G in range(16):
    open(f"{OUT}/g1_in{G}.reif", "wb").write(ExecutorInput(input_=mkinp(G), mte_tags=tags()).serialize())
    d.push(f"{OUT}/g1_in{G}.reif", f"{DEV}/g1_in{G}.reif")
su(f"chmod 666 {DEV}/g1_in*.reif")
for k, v in (("measurement_mode", "F+R"), ("pin_to_core", "8"), ("warmups", "1"),
             ("enable_pre_run_flush", "1"), ("enable_phr_flush", "0"),
             ("enable_view_rotation", "0"), ("enable_ssbs", "0")):
    su(f"echo {v} > /sys/executor/{k}")

idx = 63 - READ_SET
out(f"MAIN={hex(MAIN)} T={T} read={READ_SET} ldnum={LDNUM} gap={GAP} reps={N} core=8 (internal-loop iters sweep)")
for iters in ITERS_LIST:
    binf = build_tc(iters)
    d.push(binf, binf.replace(OUT, DEV)); su(f"chmod 666 {binf.replace(OUT, DEV)}")
    su(f"{EU} 1 >/dev/null; {EU} w {binf.replace(OUT, DEV)} >/dev/null")
    res = {}
    for G in range(16):
        o = su(f"sh {DEV}/run_tag.sh {DEV}/g1_in{G}.reif {N}")
        hs = re.findall(r"htrace 0:\s*([01]{64})", o)
        res[G] = (sum(int(h[idx]) for h in hs) / len(hs)) if hs else -1.0
    mm = [res[g] for g in range(16) if g != T and g != 15]
    sep = res[T] - max(mm)
    out(f"iters={iters:4d}  MATCH={res[T]:.2f}  checked-mm max={max(mm):.2f} mean={sum(mm)/len(mm):.2f}"
        f"  sep={sep:+.2f}  {'DETECT' if sep > 0.15 else '-'}")
