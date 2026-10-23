import struct, sys
VD = "/home/gal_k_1_1998/revizor/sca-fuzzer/docs/tiktag_mte_ni/general_detection/eval_out/collect/session_20261023_011712/violation-261023-021502"
MAGIC = 0x49525A5652
SEC_MTE_TAGS = 0x06
def sections(buf):
    magic, ver, hlen, n, flags, total = struct.unpack_from("<6Q", buf, 0)
    assert magic == MAGIC, hex(magic)
    out = {}
    for i in range(n):
        t, f, off, ln = struct.unpack_from("<4Q", buf, 48 + 32 * i)
        out[t] = buf[off:off + ln]
    return out
def mte_tags(path, count=512):
    buf = open(path, "rb").read()
    sec = sections(buf)
    p = sec[SEC_MTE_TAGS]
    tags = []
    for i in range(count):
        byte = p[i // 2]
        tags.append((byte & 0xF) if i % 2 == 0 else (byte >> 4) & 0xF)
    return tags, sorted(sec.keys())
if __name__ == "__main__":
    res = {}
    for n in (0, 8, 16, 24):
        t, secs = mte_tags(f"{VD}/input_{n:04d}.reif")
        res[n] = t
        print(f"input {n:04d}: sections={[hex(s) for s in secs]} main|faulty granules={len(t)}")
    # granules of interest (offset/16): from CE trace
    interest = {
        "arch ldr w5 @0x28 (set0)": 2, "arch stp @0x40 (set38)": 408, "arch ldr w0 @0x64 (set1)": 261,
        "arch ldr x0 @0x7c (set61)": 503, "arch ldr w3 @0x9c (set63)": 508, "arch ldr w3 @0xbc (set13)": 54,
        "arch ldr x5 @0x11c (set46) GUESS": 442, "arch stp w4 @0x13c (set20)": 339,
        "spec str @0x160 winA (set42)": 171, "spec str @0x160 winB (set17)": 69, "spec str @0x160 winC (set4)": 17,
    }
    print()
    print(f"{'granule':>8} {'role':<36} " + " ".join(f"in{n:>02d}" for n in res))
    for role, g in sorted(interest.items(), key=lambda kv: kv[1]):
        print(f"{g:>8} {role:<36} " + " ".join(f"{res[n][g]:>4d}" for n in res))
    # are memory tags identical across the four variants?
    diff = [g for g in range(512) if len({res[n][g] for n in res}) > 1]
    print(f"\ngranules whose memory tag differs across the 4 variants: {len(diff)} -> {diff[:40]}")
    from collections import Counter
    print("memory tag histogram (input 0):", sorted(Counter(res[0]).items()))
