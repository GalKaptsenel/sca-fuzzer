import sys
sys.path.insert(0, "/tmp/claude-1000/-home-gal-k-1-1998-revizor-sca-fuzzer/cf5e0fed-2c4e-46af-8f5e-49fcf726976a/scratchpad")
from ce_parse import parse, VD, CODE_BASE
n = int(sys.argv[1])
ents = parse(f"{VD}/input_{n:04d}_trace_log.txt")
for e in ents:
    off = e["pc"] - CODE_BASE
    ms = ""
    for ma in e["mem"]:
        addr = ma["ea"] & 0x00ffffffffffffff
        sb = e["x"][29] & 0x00ffffffffffffff
        o = addr - sb
        ms += f" | {ma['type']} gran{o//16}(off {o:#x}) set{(addr>>6)&63} tag{(ma['ea']>>56)&0xf} before={ma['before']:#x} after={ma['after']:#x} sz{ma['size']}"
    print(f"[{e['idx']:03d}] n{e['nest']} +{off:#05x} {e['asm']:<34s}{ms}")
