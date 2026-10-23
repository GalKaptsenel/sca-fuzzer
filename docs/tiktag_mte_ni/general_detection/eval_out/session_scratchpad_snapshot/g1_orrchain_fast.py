"""Fast serial-orr-chain G1 sweep: the full rep loop runs in ONE on-device shell per tag (run_tag.sh),
so adb carries ~100 calls total instead of thousands. Gap = N serial `orr x2,x2,x0` (x0=0 guess result,
x2=probe addr): a latency-gated probe the tag-check shrink races. Sweep short chains for the boundary
where MATCH reaches set 32 while mismatches don't. Self-inflicted single-input tag sweep."""
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
PROBE_SET = 32
LDNUM = 2
CHAINS = [0, 4, 8, 16, 32, 64, 96]
N = 40
EU = "/data/local/tmp/executor_userland /dev/executor"
DEV = "/data/local/tmp"
LOG = open(f"{OUT}/g1_orrchain_results.txt", "w")


def out(*a):
    s = " ".join(str(x) for x in a); print(s); LOG.write(s + "\n"); LOG.flush()


d = Client('127.0.0.1', 5037).devices()[0]


def su(cmd):
    return d.shell(f"su -c '{cmd}'")


MAIN = int(su("cat /sys/executor/print_sandbox_base").strip(), 16)


def gadget_asm(ldnum, chain):
    L = ["ldr x10, [x3]",            # WARM guess granule via unchecked-tag alias -> L1 (no fault)
         "dc civac, x0", "dc civac, x2", "dsb sy", "isb", "ldr x0, [x0]", "cbz x0, .Ls"]
    L += ["ldr x0, [x1]"] * ldnum    # guess load: L1 data hit + tag check (the variable)
    L += ["orr x2, x2, x0"] * chain  # serial latency gate feeding probe addr
    L += ["ldr x9, [x2]", ".Ls:", "nop"]
    return "\n".join(L) + "\n"


def build_tc(chain):
    asm = f"{OUT}/g1_or{chain}.asm"; binf = f"{OUT}/g1_or{chain}.bin"
    open(asm, "w").write(gadget_asm(LDNUM, chain))
    r = subprocess.run([PY, COMPILE, "--input", asm, "--output", binf], capture_output=True)
    if r.returncode:
        out(r.stderr.decode()); sys.exit(1)
    return binf


def mkinp(G):
    inp = Input(1); inp.view('uint64')[:] = 0
    g = inp[0]['gpr']
    g[0] = MAIN + 0x400
    g[1] = (MAIN & ~(0xF << 56)) | (G << 56)       # guess ptr, swept tag G (checked)
    g[2] = MAIN + PROBE_SET * 64                    # probe ptr (set 32)
    g[3] = (MAIN & ~(0xF << 56)) | (15 << 56)       # warm alias to guess granule, unchecked tag 15
    return inp


def tags():
    t = [15] * 512; t[0] = T; return t


# push the on-device runner + per-tag REIFs
d.push(f"{OUT}/run_tag.sh", f"{DEV}/run_tag.sh"); su(f"chmod 755 {DEV}/run_tag.sh")
for G in range(16):
    ei = ExecutorInput(input_=mkinp(G), mte_tags=tags())
    open(f"{OUT}/g1_in{G}.reif", "wb").write(ei.serialize())
    d.push(f"{OUT}/g1_in{G}.reif", f"{DEV}/g1_in{G}.reif")
su(f"chmod 666 {DEV}/g1_in*.reif")

for k, v in (("measurement_mode", "F+R"), ("warmups", "1"),
             ("enable_pre_run_flush", "0"), ("enable_phr_flush", "0"),
             ("enable_view_rotation", "0"), ("reload_isolate", "0"),
             ("reload_random_order", "0"), ("enable_ssbs", "0")):
    su(f"echo {v} > /sys/executor/{k}")
su("echo -1 > /sys/executor/pin_to_core")   # no pinning (runs on current CPU)

idx = 63 - PROBE_SET
out(f"MAIN={hex(MAIN)} T={T} probe set={PROBE_SET} ldnum={LDNUM} reps={N} (batched on-device)")
for chain in CHAINS:
    binf = build_tc(chain)
    d.push(binf, binf.replace(OUT, DEV)); su(f"chmod 666 {binf.replace(OUT, DEV)}")
    su(f"{EU} 1 >/dev/null; {EU} w {binf.replace(OUT, DEV)} >/dev/null")
    res = {}
    for G in range(16):
        o = su(f"sh {DEV}/run_tag.sh {DEV}/g1_in{G}.reif {N}")
        hs = re.findall(r"htrace 0:\s*([01]{64})", o)
        res[G] = (sum(int(h[idx]) for h in hs) / len(hs)) if hs else -1.0
    mm = [res[g] for g in range(16) if g != T]
    sep = res[T] - max(mm)
    out(f"--- chain={chain} ---")
    out(" ".join(f"{g}:{res[g]:.2f}" for g in range(16)))
    out(f"   MATCH={res[T]:.2f} mm-mean={sum(mm)/len(mm):.2f} mm-max={max(mm):.2f} sep={sep:+.2f}"
        f" {'*** SEPARATED ***' if sep > 0.15 else ''}")
