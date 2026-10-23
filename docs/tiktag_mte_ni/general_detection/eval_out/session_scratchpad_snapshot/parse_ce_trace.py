import re
import sys

LOW56 = (1 << 56) - 1


def parse(path):
    entries, cur = [], None
    for raw in open(path).read().splitlines():
        s = raw.strip()
        m = re.match(r"--- Entry (\d+) ---$", s)
        if m:
            if cur is not None:
                entries.append(cur)
            cur = {"idx": int(m.group(1))}
            continue
        if cur is None:
            continue
        m = re.match(r"x29 : ?(0x[0-9a-f]+)$", s) or re.match(r"x29: (0x[0-9a-f]+)$", s)
        if m:
            cur["x29"] = int(m.group(1), 16)
        m = re.match(r"PC : (0x[0-9a-f]+)$", s)
        if m:
            cur["pc"] = int(m.group(1), 16)
        m = re.match(r"Encoding: (0x[0-9a-f]+) \((.*)\)$", s)
        if m:
            cur["asm"] = m.group(2)
        m = re.match(r"Speculation nesting: (\d+)$", s)
        if m:
            cur["nest"] = int(m.group(1))
        m = re.match(r"Has memory access: (\d)$", s)
        if m:
            cur["has_mem"] = int(m.group(1))
        m = re.match(r"EA: (0x[0-9a-f]+)$", s)
        if m:
            cur["ea"] = int(m.group(1), 16)
        m = re.match(r"Type: (\w+)$", s)
        if m:
            cur["type"] = m.group(1)
        m = re.match(r"Size: (\d+)$", s)
        if m:
            cur["size"] = int(m.group(1))
    if cur is not None:
        entries.append(cur)
    return entries


def main():
    path = sys.argv[1]
    es = parse(path)
    code_base = es[0]["pc"]
    base = es[0]["x29"]
    print(f"file={path}")
    print(f"entries={len(es)} code_base=0x{code_base:x} x29=0x{base:x} x29_low56_off={base & LOW56 & 0xfff:#x}")
    print("nest_hist:", {n: sum(1 for e in es if e.get('nest') == n) for n in sorted({e.get('nest') for e in es})})
    print("--- memory accesses (entry, pc_off, nesting, type, size, ptr_tag, offset_from_x29, granule, set, asm) ---")
    for e in es:
        if e.get("has_mem") != 1:
            continue
        ea = e["ea"]
        tag = (ea >> 56) & 0xF
        off = (ea & LOW56) - (e["x29"] & LOW56)
        g = off // 16
        st = (off >> 6) & 63
        print(f"  [{e['idx']:03d}] off=0x{e['pc']-code_base:03x} nest={e['nest']} {e.get('type')} sz={e.get('size')} "
              f"ptrtag={tag:2d} off=0x{off:x} gran={g} set={st}  {e.get('asm')}")
    print("--- instruction listing (entry, pc_off, nest, asm) ---")
    for e in es:
        print(f"  [{e['idx']:03d}] 0x{e['pc']-code_base:03x} n={e['nest']} {e.get('asm')}")


if __name__ == "__main__":
    main()
