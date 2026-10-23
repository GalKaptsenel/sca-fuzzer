import re
import sys

text = open(sys.argv[1]).read().splitlines()


def block(inp_no):
    rows, on = [], False
    for ln in text:
        if re.match(rf"^Input #{inp_no}$", ln.strip()):
            on = True
            continue
        if on and ln.startswith("* Contract trace"):
            break
        if on:
            m = re.match(r"^\s*([\^\.]{64})\s+\[(\d+)\]\s*$", ln)
            if m:
                rows.append((m.group(1), int(m.group(2))))
    return rows


def residency(rows):
    tot = sum(c for _, c in rows)
    res = {}
    for s in range(64):
        idx = 63 - s  # report string is MSB-first: set = 63 - char index
        res[s] = sum(c for st, c in rows if st[idx] == "^") / tot
    return res, tot


r0 = block(0)
r8 = block(8)
res0, t0 = residency(r0)
res8, t8 = residency(r8)
print(f"input#0 rows={len(r0)} total_samples={t0}; input#8 rows={len(r8)} total_samples={t8}")
watch = [0, 1, 4, 6, 7, 8, 13, 17, 20, 32, 38, 42, 46, 52, 61, 63]
print("set   #0     #8     |diff|")
for s in watch:
    print(f"{s:3d}  {res0[s]:.3f}  {res8[s]:.3f}  {abs(res0[s]-res8[s]):.3f}")
diff = sorted([(abs(res0[s] - res8[s]), s, res0[s], res8[s]) for s in range(64)], reverse=True)
print("top diverging sets (|d|>=0.05):", [(s, round(a, 3), round(b, 3)) for a, s, b, _ in diff if a >= 0.05])
print("set 52 residency #0=%.3f #8=%.3f" % (res0[52], res8[52]))
