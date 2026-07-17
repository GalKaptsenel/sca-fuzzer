"""Load a raw hand-assembled TC + one input on the phone, F+R measure it N times, report set freqs."""
import os, re, sys, subprocess
sys.path.insert(0, '/home/gal_k_1_1998/sca-fuzzer')
from src.aarch64.aarch64_connection import ADBConnection

EU   = '/data/local/tmp/executor_userland /dev/executor'
SYSF = '/sys/executor'
REIF = '/home/gal_k_1_1998/sca-fuzzer/violations_fr/violation-260716-181940/input_0009.reif'
conn = ADBConnection('127.0.0.1', 5037)

def sh(c):
    return conn.shell(c, privileged=True)

def setup():
    sh(f'echo "F+R" > {SYSF}/measurement_mode')
    sh(f'echo 0 > {SYSF}/enable_pre_run_flush')
    conn.push(REIF, '/data/local/tmp/pf_in.reif')

def load(tcbin):
    conn.push(tcbin, '/data/local/tmp/pf_tc.bin')
    sh(f'{EU} 9')                                     # CLEAR_ALL_INPUTS
    out = sh(f'{EU} 5')                               # ALLOCATE_INPUT
    m = re.search(r'Allocated Input ID:\s*(\d+)', out, re.I)
    iid = int(m.group(1))
    sh(f'{EU} 4 {iid}')                               # CHECKOUT_INPUT
    sh(f'{EU} w /data/local/tmp/pf_in.reif')
    sh(f'{EU} 1')                                     # CHECKOUT_TEST
    sh(f'{EU} w /data/local/tmp/pf_tc.bin')
    return iid

def measure(iid, n):
    freq = [0]*64
    got = 0
    for _ in range(n):
        sh(f'{EU} 8')                                 # TRACE
        sh(f'{EU} 4 {iid}')                           # CHECKOUT_INPUT
        out = sh(f'{EU} 7')                           # MEASUREMENT
        m = re.search(r'htrace 0:\s*([01]{64})', out, re.I)
        if not m:
            continue
        s = m.group(1); got += 1
        for pos, ch in enumerate(s):
            if ch == '1':
                freq[63-pos] += 1                     # set = 63 - position
    return [f/max(got,1) for f in freq], got

def run(tcbin, tag, n):
    iid = load(tcbin)
    fr, got = measure(iid, n)
    lit = [b for b in range(64) if fr[b] > 0.5]
    print(f"  [{tag:26}] reps={got:3}  lit={lit}")
    return fr

if __name__ == '__main__':
    setup()
    print("=== smoke: demand-load faulty 0x1000 + flush the demand line ===")
    run(f'{os.path.dirname(__file__)}/t_demand_only.bin', 'V=0x1000 demand-flushed', 30)
