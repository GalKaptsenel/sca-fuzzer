import re
import sys

LOW56 = (1 << 56) - 1
text = open(sys.argv[1]).read().splitlines()
want = {int(x) for x in sys.argv[2].split(",")}
cur, ent = None, {}
for ln in text:
    m = re.match(r"--- Entry (\d+) ---$", ln.strip())
    if m:
        cur = int(m.group(1))
        continue
    if cur in want:
        m = re.match(r"(x\d\d|SP) ?: (0x[0-9a-f]+)$", ln.strip())
        if m:
            ent.setdefault(cur, {})[m.group(1)] = int(m.group(2), 16)
x29 = None
for i in sorted(want):
    r = ent.get(i, {})
    print(f"entry {i}: " + " ".join(f"{k}=0x{v:x}" for k, v in sorted(r.items()) if k in ("x00", "x01", "x02", "x03", "x04", "x29")))
