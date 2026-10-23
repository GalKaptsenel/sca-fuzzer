"""Faithful official-G1 gap: the 'gap' is N serial `orr x2, x2, x0` ops (x0 = the zeroed guess-load
result, x2 = the probe address), i.e. a value-stable but serially latency-bound chain that the probe
address depends on -- the probe cannot issue until the ~N-cycle chain finishes. A tag mismatch shrinks
the speculative window; if N is tuned so the shrunk window no longer covers the chain, the mismatch
probe never issues (set 32 cold) while the match probe does (set 32 hot). Single run (control E showed
single-run forward-cbz speculation reaches a probe at ~0.9), self-inflicted tag sweep. Sweep N around 80."""
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
CHAINS = [0, 4, 8, 12, 16, 24]
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
    L = ["dc civac, x0", "dc civac, x2", "dsb sy", "isb",
         "ldr x0, [x0]",            # slow load; *slow=0 -> x0=0, line flushed -> long window
         "cbz x0, .Ls"]            # forward, arch-taken; spec fall-through
    L += ["ldr x0, [x1]"] * ldnum   # tag-checked guess load(s); x0 = *(guess) = 0 on success
    L += ["orr x2, x2, x0"] * chain  # serial latency chain feeding probe addr x2 (x0=0 -> x2 stable)
    L += ["ldr x9, [x2]", ".Ls:", "nop"]   # probe: issues only if window covered the chain
    return "\n".join(L) + "\n"


def build_tc(chain):
    asm = f"{OUT}/g1_or{chain}.asm"; binf = f"{OUT}/g1_or{chain}.bin"
    open(asm, "w").write(gadget_asm(LDNUM, chain))
    r = subprocess.run([PY, COMPILE, "--input", asm, "--output", binf], capture_output=True)
    if r.returncode:
        out(r.stderr.decode()); sys.exit(1)
    return binf


def mkinp(G):
    inp = Input(1); inp.view('uint64')[:] = 0        # all memory 0 -> guess granule content = 0
    g = inp[0]['gpr']
    g[0] = MAIN + 0x400                               # x0 slow/cond ptr (set 16)
    g[1] = (MAIN & ~(0xF << 56)) | (G << 56)          # x1 guess ptr off0 (set 0), swept tag G
    g[2] = MAIN + PROBE_SET * 64                      # x2 probe ptr (set 32)
    return inp


def tags():
    t = [15] * 512; t[0] = T; return t                # guess granule 0 = checked tag T


for G in range(16):
    ei = ExecutorInput(input_=mkinp(G), mte_tags=tags())
    open(f"{OUT}/g1_in{G}.reif", "wb").write(ei.serialize())
    d.push(f"{OUT}/g1_in{G}.reif", f"{DEV}/g1_in{G}.reif")
su(f"chmod 666 {DEV}/g1_in*.reif")

for k, v in (("measurement_mode", "F+R"), ("pin_to_core", "8"),
             ("warmups", "1"), ("enable_pre_run_flush", "1"), ("enable_ssbs", "0")):
    su(f"echo {v} > /sys/executor/{k}")


def htrace_bit(o, bit):
    m = re.search(r"htrace 0:\s*([01]{64})", o)
    return int(m.group(1)[63 - bit]) if m else None


out(f"MAIN={hex(MAIN)} T={T} probe set={PROBE_SET} ldnum={LDNUM} reps={N}")
for chain in CHAINS:
    binf = build_tc(chain)
    d.push(binf, binf.replace(OUT, DEV)); su(f"chmod 666 {binf.replace(OUT, DEV)}")
    su(f"{EU} 1 >/dev/null; {EU} w {binf.replace(OUT, DEV)} >/dev/null")
    res = {}
    for G in range(16):
        su(f"{EU} 9 >/dev/null")
        iid = re.search(r"Allocated Input ID:\s*(\d+)", su(f"{EU} 5")).group(1)
        su(f"{EU} 4 {iid} >/dev/null; {EU} w {DEV}/g1_in{G}.reif >/dev/null")
        hits = 0
        for _ in range(N):
            su(f"{EU} 8 >/dev/null")
            b = htrace_bit(su(f"{EU} 4 {iid} >/dev/null; {EU} 7"), PROBE_SET)
            if b is not None:
                hits += b
        res[G] = hits / N
    mm = [res[g] for g in range(16) if g != T]
    sep = res[T] - max(mm)
    out(f"--- chain={chain} ---")
    out(" ".join(f"{g}:{res[g]:.2f}" for g in range(16)))
    out(f"   MATCH={res[T]:.2f} mm-mean={sum(mm)/len(mm):.2f} mm-max={max(mm):.2f} sep={sep:+.2f}"
        f" {'*** SEPARATED ***' if sep > 0.15 else ''}")
