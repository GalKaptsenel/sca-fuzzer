"""Durable, deterministic investigation harness for the regular-sealed PAC×MTE WOULD-FPAC.
Lives in /home (survives the VM's spontaneous reboots; /tmp does not).

Per trial: random.seed(trial) -> deterministic gen/input seeds -> reproducible TCs. Wraps the
CE-trace chokepoint _seal_trace: clears dmesg before each genuine trace, checks after. On the first
WOULD-FPAC it writes a COMPLETE report (trial seed for exact repro, every dmesg line, every
[SEAL-PAC] sign-time log captured this resolve, the correlation auth_sig<->seal_sig with a bitwise
diff, the seal sites, and the full sealed-TC asm) to LOGFILE and stdout, then exits.

Run:  revizor-venv/bin/python investigate_seal_fpac.py [N_trials]
"""
import os, sys, subprocess, random, io, contextlib, re
sys.path.insert(0, '/home/gal_k_1_1998/sca-fuzzer'); os.chdir('/home/gal_k_1_1998/sca-fuzzer')

from src.config import CONF
CONF.load('config_pac_mte_basic.yml')
from src.isa_loader import InstructionSet
from src.aarch64.aarch64_generator import Aarch64RandomGenerator
from src.aarch64.aarch64_target_desc import Aarch64TargetDesc
from src.aarch64.aarch64_printer import Aarch64Printer, Aarch64ASMLayout
from src.aarch64.aarch64_kernel import PacKeys
from src.aarch64.aarch64_executor import Aarch64NonInterferenceExecutor as Base
from src import factory

LOGFILE = '/home/gal_k_1_1998/seal_fpac_report.txt'
isa = InstructionSet('base.json', CONF.instruction_categories)
td  = Aarch64TargetDesc()

def dmesg_clear(): subprocess.run(['sudo', 'dmesg', '-C'], stderr=subprocess.DEVNULL)
def dmesg_fpac():
    out = subprocess.run(['sudo', 'dmesg'], capture_output=True, text=True).stdout
    return [l.split('] ', 1)[-1] for l in out.splitlines() if 'would FPAC' in l]

# capture every [SEAL-PAC] line emitted by _resolve_pac during the current resolve
_seal_pac_buf = []
class _Tee(io.TextIOBase):
    def __init__(self, real): self.real = real
    def write(self, s):
        if '[SEAL-PAC]' in s: _seal_pac_buf.append(s.strip())
        return self.real.write(s)
sys.stderr = _Tee(sys.stderr)

def report(trial, gen_seed, inp, fpac, asm, sealed):
    out = []
    def w(s): out.append(s)
    w("="*100)
    w(f"WOULD-FPAC reproduced.  REPRO: random.seed({trial}); gen_seed={gen_seed}; inp.seed={getattr(inp,'seed','?')}")
    w(f"sealer={type(sealed).__name__}")
    w("--- dmesg WOULD-FPAC (auth-time ptr/ctx) ---")
    for l in fpac: w("  " + l)
    w(f"--- [SEAL-PAC] sign-time logs this resolve ({len(_seal_pac_buf)}) ---")
    for l in _seal_pac_buf: w("  " + l)
    w("--- correlation (match auth sig=(ptr>>48) to a seal sig; bitwise diff) ---")
    for f in fpac:
        m = re.search(r'op=(\d+) ptr=0x([0-9a-f]+) ctx=0x([0-9a-f]+)', f)
        if not m: continue
        op = int(m.group(1)); aptr = int(m.group(2), 16); actx = int(m.group(3), 16)
        asig = (aptr >> 48) & 0xffff
        w(f"  AUTH op={op} ptr={aptr:#018x} ctx={actx:#018x}  auth_sig={asig:#06x}")
        for s in _seal_pac_buf:
            sm = re.search(r'sig=0x([0-9a-f]+) signed_ptr=0x([0-9a-f]+) cval=0x([0-9a-f]+)', s)
            if not sm: continue
            if int(sm.group(1), 16) != asig: continue
            sp = int(sm.group(2), 16); cv = int(sm.group(3), 16)
            w(f"    MATCH seal: {s.split('[SEAL-PAC] ',1)[-1]}")
            w(f"      ptr ^ signed_ptr = {aptr ^ sp:#018x}   ctx ^ cval = {actx ^ cv:#018x}")
            w(f"      tag[59:56]: auth={(aptr>>56)&0xff:#04x} signed={(sp>>56)&0xff:#04x}  "
              f"addr[47:0]: auth={aptr&((1<<48)-1):#014x} signed={sp&((1<<48)-1):#014x}")
    w("--- PAC seal sites ---")
    for s in getattr(sealed, '_pac', []):
        w("  " + str({k: v for k, v in vars(s).items() if k in ('value_reg', 'slot_locs')}))
    w("--- MTE seal sites ---")
    for s in getattr(sealed, '_mte', []):
        w("  " + str({k: v for k, v in vars(s).items() if k in ('value_reg', 'slot_locs')}))
    w("--- sealed TC asm ---")
    w(asm)
    text = "\n".join(out)
    open(LOGFILE, 'w').write(text)
    print(text); sys.stdout.flush()

_orig = Base._seal_trace
def _wrapped(self, tc, inp):
    dmesg_clear()
    r = _orig(self, tc, inp)
    hits = dmesg_fpac()
    if hits:
        asm = Aarch64Printer(td).print_layout(Aarch64ASMLayout(tc))
        report(getattr(self, '_dbg_trial', -1), getattr(self, '_dbg_genseed', -1), inp, hits, asm, self._sealed)
        os._exit(0)
    return r
Base._seal_trace = _wrapped

def make_exec(gen_seed):
    gen = Aarch64RandomGenerator(isa, gen_seed)
    ex = factory.get_executor(generator=gen)
    k = PacKeys()
    k.apia_lo = k.apib_lo = k.apda_lo = k.apdb_lo = k.apga_lo = 0x1122334455667788
    k.apia_hi = k.apib_hi = k.apda_hi = k.apdb_hi = k.apga_hi = 0x8877665544332211
    ex.local_executor.set_pac_keys(k)
    return ex, gen

import tempfile
N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
tmp = tempfile.mkdtemp()
for trial in range(N):
    random.seed(trial)
    gen_seed = random.randrange(1 << 32)
    ig_seed  = random.randrange(1 << 32)
    ex, gen = make_exec(gen_seed)
    ex._dbg_trial = trial; ex._dbg_genseed = gen_seed
    ig = factory.get_input_generator(ig_seed)
    tc = None
    for _ in range(20):
        try:
            tc = gen.create_test_case(os.path.join(tmp, 't.asm'), disable_assembler=True)
        except Exception:
            continue
        ex.load_test_case(tc)
        if getattr(ex._sealed, '_pac', []):
            break
        tc = None
    if tc is None:
        continue
    ex._sandbox_base, _ = ex.read_base_addresses()
    for inp in ig.generate(3):
        _seal_pac_buf.clear()
        try:
            ex.trace_test_case_with_taints([inp], CONF.model_max_nesting)
        except SystemExit:
            raise
        except Exception:
            pass
    if trial % 20 == 0:
        print(f"trial {trial}: no FPAC yet", flush=True)
print("NO HIT in", N, "trials")
