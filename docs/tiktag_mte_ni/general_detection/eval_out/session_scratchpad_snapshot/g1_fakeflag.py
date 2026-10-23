"""Prove the warm-up-flag idea through the raw executor (no kernel change): a flag in sandbox memory
decides, per run, whether the gadget TRAINS (branch falls through, genuine/match tag, warm set) or
ATTACKS (branch flips, swept tag, read set). We drive K training runs then one attack run with BPU
state preserved (bt2.sh), sweep the attack tag, and measure only the attack run's read set. A FAST
selector (cached flag cell) steers the register/probe choice so the speculative body's addresses are
ready; a SLOW condition (flushed flag cell) opens the window. If the match tag separates, the flag
schedule reproduces v1 through the executor's per-run path -> justifies the executor warm-up-flag change."""
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
T, READ_SET, WARM_SET, LDNUM, GAP, K, R = 5, 32, 40, 16, 80, 32, 12
SEL_OFF, COND_OFF = 0xC00, 0xE00     # fast selector cell (set 48), slow condition cell (set 56)
DEV = "/data/local/tmp"
EU = f"{DEV}/executor_userland /dev/executor"
d = Client('127.0.0.1', 5037).devices()[0]


def su(c):
    return d.shell(f"su -c '{c}'")


MAIN = int(su("cat /sys/executor/print_sandbox_base").strip(), 16)


def gadget_asm():
    L = ["ldr x7, [x0]",            # FAST selector: 1=train, 0=attack (x0 cell stays cached)
         "cmp x7, #0",
         "csel x6, x5, x1, eq",     # attack(0)->swept x5 ; train(1)->match x1
         "csel x10, x2, x4, eq",    # attack(0)->read set x2 ; train(1)->warm set x4
         "dc civac, x3",            # flush the SLOW condition cell -> long window
         "dc civac, x2", "dsb sy", "isb",
         "ldr x8, [x3]",            # SLOW condition: 1=train (fall through), 0=attack (taken)
         "cbz x8, .Lskip"]          # train: not-taken->body(arch, trains); attack: taken->skip, spec->body
    L += ["ldr x9, [x6]"] * LDNUM + ["orr x11, xzr, xzr"] + ["orr x10, x10, x11"] * GAP
    L += ["ldr x12, [x10]", ".Lskip:"]
    return "\n".join(L) + "\n"


def build_tc():
    asm = f"{OUT}/g1_ff.asm"; binf = f"{OUT}/g1_ff.bin"
    open(asm, "w").write(gadget_asm())
    r = subprocess.run([PY, COMPILE, "--input", asm, "--output", binf], capture_output=True)
    if r.returncode:
        print(r.stderr.decode()); sys.exit(1)
    return binf


def mk(flag, G):
    inp = Input(1); inp.view('uint64')[:] = 0
    g = inp[0]['gpr']
    g[0] = MAIN + SEL_OFF; inp[0]['main'][SEL_OFF // 8] = flag     # fast selector cell
    g[3] = MAIN + COND_OFF; inp[0]['main'][COND_OFF // 8] = flag   # slow condition cell
    g[1] = (MAIN & ~(0xF << 56)) | (T << 56)                       # match guess ptr (training)
    g[5] = (MAIN & ~(0xF << 56)) | (G << 56)                       # swept guess ptr (attack)
    g[2] = MAIN + READ_SET * 64                                    # read set probe (attack)
    g[4] = MAIN + WARM_SET * 64                                    # warm set probe (training)
    t = [15] * 512; t[0] = T
    return ExecutorInput(input_=inp, mte_tags=t).serialize()


binf = build_tc()
d.push(binf, binf.replace(OUT, DEV)); su(f"chmod 666 {binf.replace(OUT, DEV)}")
d.push(f"{OUT}/bt2.sh", f"{DEV}/bt2.sh"); su(f"chmod 755 {DEV}/bt2.sh")
for G in range(16):
    open(f"{OUT}/ff_atk{G}.reif", "wb").write(mk(0, G))   # attack: flag=0
    d.push(f"{OUT}/ff_atk{G}.reif", f"{DEV}/ff_atk{G}.reif")
open(f"{OUT}/ff_train.reif", "wb").write(mk(1, T))        # training: flag=1 (G unused on training)
d.push(f"{OUT}/ff_train.reif", f"{DEV}/ff_train.reif")
su(f"chmod 666 {DEV}/ff_*.reif")

# preserve BPU across runs: no phr flush / view rotation; gadget self-flushes its lines
for k, v in (("measurement_mode", "F+R"), ("pin_to_core", "8"), ("warmups", "0"),
             ("enable_pre_run_flush", "0"), ("enable_phr_flush", "0"),
             ("enable_view_rotation", "0"), ("enable_ssbs", "0")):
    su(f"echo {v} > /sys/executor/{k}")
su(f"{EU} 1 >/dev/null; {EU} w {binf.replace(OUT, DEV)} >/dev/null")

idx = 63 - READ_SET
print(f"MAIN={hex(MAIN)} T={T} read={READ_SET} K-train={K} R={R} ldnum={LDNUM} gap={GAP} core=8 (fake warm-up flag)")
res = {}
for G in range(16):
    o = su(f"sh {DEV}/bt2.sh {DEV}/ff_train.reif {DEV}/ff_atk{G}.reif {K} {R}")
    hs = re.findall(r"htrace 0:\s*([01]{64})", o)
    res[G] = (sum(int(h[idx]) for h in hs) / len(hs)) if hs else -1.0
    print(f"{G:2d} {'MATCH ' if G == T else 'mismis'} {res[G]:.2f}", flush=True)

mm = [res[g] for g in range(16) if g != T and g != 15]
sep = res[T] - max(mm)
print(f"\nMATCH(T={T})={res[T]:.2f}  checked-mm max={max(mm):.2f} mean={sum(mm)/len(mm):.2f} sep={sep:+.2f}"
      f"  {'*** fake warm-up flag REPRODUCES v1 ***' if sep > 0.15 else 'no separation'}")
