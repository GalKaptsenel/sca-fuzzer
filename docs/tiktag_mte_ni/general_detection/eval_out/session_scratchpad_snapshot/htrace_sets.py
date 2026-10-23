import re, sys
from collections import defaultdict

rep = open(sys.argv[1]).read()
# split per Input block
blocks = re.split(r'Input #(\d+)\n\* Hardware trace:\n', rep)
# blocks[0]=preamble, then pairs (num, body)
data = {}
for i in range(1, len(blocks), 2):
    num = int(blocks[i]); body = blocks[i+1]
    freq = defaultdict(float); total = 0
    for line in body.splitlines():
        m = re.match(r'\s*([\^\.]+)\s+\[(\d+)\]', line)
        if not m: continue
        s, c = m.group(1), int(m.group(2))
        total += c
        for idx, ch in enumerate(s):
            if ch == '^':
                setnum = 63 - idx          # MSB-first: char idx 0 == set 63
                freq[setnum] += c
    if total:
        data[num] = ({k: v/total for k, v in freq.items()}, total, len(body.splitlines()[0].strip().split()[0]))

f0 = data[0][0]; f8 = data[8][0]
print(f"input0 strlen={data[0][2]}  input8 strlen={data[8][2]}")
print(f"\n{'set':>3} {'in0':>6} {'in8':>6} {'|delta|':>7}  note")
allsets = sorted(set(f0)|set(f8))
CE_arch=[0,1,13,20,38,46,61,63]; CE_spec=[4,17,42]
for s in allsets:
    a=f0.get(s,0.0); b=f8.get(s,0.0); d=abs(a-b)
    note=[]
    if s in CE_arch: note.append("CE-arch")
    if s in CE_spec: note.append("CE-spec-only")
    if s==52: note.append("<<AUTHOR set52")
    if s in (6,7,8,32): note.append("AUTO_TRIAGE diverging")
    flag = "  *DIVERGE*" if d>=0.08 else ""
    print(f"{s:>3} {a:>6.3f} {b:>6.3f} {d:>7.3f}  {' '.join(note)}{flag}")
print(f"\nHW diverging (|d|>=0.08): {sorted(s for s in allsets if abs(f0.get(s,0)-f8.get(s,0))>=0.08)}")
print(f"set52: in0={f0.get(52,0):.3f} in8={f8.get(52,0):.3f}  -> diverges? {abs(f0.get(52,0)-f8.get(52,0))>=0.08}")
print(f"set46(CE arch demand): in0={f0.get(46,0):.3f} in8={f8.get(46,0):.3f}")
