"""Fast cross-input training test: load K genuine-tag training inputs ONCE, then per swept tag append
one attack input, trace the whole unflushed batch, and measure only the attack input's probe set."""
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
T, READ_SET, WARM_SET, LDNUM, GAP, K, N = 5, 32, 48, 16, 80, 64, 30
DEV = "/data/local/tmp"
EU = f"{DEV}/executor_userland /dev/executor"
d = Client('127.0.0.1', 5037).devices()[0]


def su(c):
    return d.shell(f"su -c '{c}'")


MAIN = int(su("cat /sys/executor/print_sandbox_base").strip(), 16)


def gadget_asm():
    L = ["ldr x12, [x4]", "dc civac, x0", "dc civac, x2", "dsb sy", "isb",
         "ldr x8, [x0]", "cbz x8, .Lskip"]
    L += ["ldr x9, [x1]"] * LDNUM + ["orr x6, xzr, xzr"] + ["orr x2, x2, x6"] * GAP
    L += ["ldr x11, [x2]", ".Lskip:"]
    return "\n".join(L) + "\n"


def mk(cond_val, guess_tag, probe_set):
    inp = Input(1); inp.view('uint64')[:] = 0
    g = inp[0]['gpr']
    g[0] = MAIN + 0x400
    inp[0]['main'][0x400 // 8] = cond_val
    g[1] = (MAIN & ~(0xF << 56)) | (guess_tag << 56)
    g[2] = MAIN + probe_set * 64
    g[4] = (MAIN & ~(0xF << 56)) | (15 << 56)
    t = [15] * 512; t[0] = T
    return ExecutorInput(input_=inp, mte_tags=t).serialize()


binf = f"{OUT}/g1_bt.bin"
open(f"{OUT}/g1_bt.asm", "w").write(gadget_asm())
r = subprocess.run([PY, COMPILE, "--input", f"{OUT}/g1_bt.asm", "--output", binf], capture_output=True)
if r.returncode:
    print(r.stderr.decode()); sys.exit(1)
d.push(binf, f"{DEV}/g1_bt.bin"); su(f"chmod 666 {DEV}/g1_bt.bin")
open(f"{OUT}/g1_train.reif", "wb").write(mk(1, T, WARM_SET))        # training: fall-through, match, warm set
d.push(f"{OUT}/g1_train.reif", f"{DEV}/g1_train.reif")
for G in range(16):
    open(f"{OUT}/g1_atk{G}.reif", "wb").write(mk(0, G, READ_SET))   # attack: taken, swept tag, read set
    d.push(f"{OUT}/g1_atk{G}.reif", f"{DEV}/g1_atk{G}.reif")
for s in ("bt_setup.sh", "bt_run.sh"):
    d.push(f"{OUT}/{s}", f"{DEV}/{s}"); su(f"chmod 755 {DEV}/{s}")
su(f"chmod 666 {DEV}/g1_*.reif")

for k, v in (("measurement_mode", "F+R"), ("pin_to_core", "8"), ("warmups", "0"),
             ("enable_pre_run_flush", "0"), ("enable_phr_flush", "0"),
             ("enable_view_rotation", "0"), ("enable_ssbs", "0")):
    su(f"echo {v} > /sys/executor/{k}")
su(f"{EU} 1 >/dev/null; {EU} w {DEV}/g1_bt.bin >/dev/null")
print("setup:", su(f"sh {DEV}/bt_setup.sh {DEV}/g1_train.reif {K}").strip())

idx = 63 - READ_SET
print(f"MAIN={hex(MAIN)} T={T} read={READ_SET} K={K} ldnum={LDNUM} gap={GAP} reps={N} core=8")
res = {}
for G in range(16):
    o = su(f"sh {DEV}/bt_run.sh {DEV}/g1_atk{G}.reif {N}")
    hs = re.findall(r"htrace 0:\s*([01]{64})", o)
    res[G] = (sum(int(h[idx]) for h in hs) / len(hs)) if hs else -1.0
    print(f"{G:2d} {'MATCH ' if G == T else 'mismis'} {res[G]:.2f}", flush=True)

mm = [res[g] for g in range(16) if g != T and g != 15]
sep = res[T] - max(mm)
print(f"\nMATCH(T={T})={res[T]:.2f}  checked-mismatch max={max(mm):.2f} mean={sum(mm)/len(mm):.2f}"
      f"  sep={sep:+.2f}  {'*** cross-input training REPRODUCES v1 ***' if sep > 0.15 else 'no separation'}")
