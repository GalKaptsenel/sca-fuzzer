#!/usr/bin/env python3
"""Cache side-channel game on the AArch64 Revizor executor -- three modes.

The gadget does an unconditional load `[x1]` (stays resident -> L1) and a conditional load `[x0]` it
then `dc civac`-flushes. After the run `[x0]` is evicted from L1 but the hardware refetches it into L2.
So an F+R reload thresholded on **L2D_CACHE_REFILL** (delta==0 => line in L2), run in **per-set
isolation** (re-run the test per set; no reload-sweep self-prefetch), reports `[x0]`'s cache set in the
htrace. `[x1]` is a fixed L1 anchor; the set that `[x0]` lands in is the channel symbol.

  id       host secretly picks one of N candidate [x0] sets; recover it from the trace (no peeking).
  hard     same, but the candidates are ADJACENT sets (a single cache line apart).
  message  encode a text string across the sets (a byte = two 4-bit nibble symbols) and recover it.

Recovery only ever sees the PUBLIC alphabet + the trace -- never the secret / plaintext.

Requires the instrumented kernel module (sysfs reload_pmu_event / reload_isolate) and the barriered
test-case binary. Examples:
    python3 cache_game.py id --candidates 3,9,15,22,30,41,50,58 --rounds 12
    python3 cache_game.py hard --base 20 --span 6 --rounds 12
    python3 cache_game.py message --message "REVIZOR N3" --traces 3
"""
import os
import sys
import argparse
import random
import subprocess
import tempfile

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "src", "executor_userland", "input_generator"))
import generate_reif_input as gri  # noqa: E402  (reuses the real REIF encoder)

DEVICE = "/dev/executor"
EXE = os.path.join(_ROOT, "src", "executor_userland", "executor_userland")
TESTCASE = os.path.join(_ROOT, "src", "executor_userland", "asm_compiler", "sandboxed_test_case.bin")
SYSFS = "/sys/executor"
FAULTY_BASE = 4096   # [x0] set s -> faulty-page offset FAULTY_BASE + s*64
CLEAR, CHECKOUT_TEST, ALLOC, CHECKOUT_INPUT, TRACE, MEASURE = 9, 1, 5, 4, 8, 7


def _exe(*args):
    return subprocess.run([EXE, DEVICE, *map(str, args)], capture_output=True, text=True).stdout


def _write_sysfs(name, value):
    path = os.path.join(SYSFS, name)
    if not os.path.exists(path):
        sys.exit(f"[!] {path} missing -- needs the instrumented kernel module (reload_* knobs).")
    with open(path, "w") as f:
        f.write(str(value))


def configure_executor():
    """F+R + L2D residency + per-set isolation + SSB mitigation on."""
    for name, val in (("measurement_mode", "F+R"), ("reload_pmu_event", "0x17"),
                      ("reload_isolate", "1"), ("reload_timer_shift", "-1"),
                      ("reload_random_order", "0"), ("enable_ssbs", "0")):
        _write_sysfs(name, val)


class Channel:
    """Transmits a symbol = the cache set that the gadget's flushed [x0] lands in."""

    def __init__(self, anchor_set, seed):
        self.anchor = anchor_set
        self.rng = random.Random(seed)
        self.seed = seed
        self.tmp = tempfile.mkdtemp(prefix="cache_game_")
        # one random base input I; only [x0] changes per transmitted symbol
        self.base = {
            "x1": anchor_set * 64, "x2": self.rng.getrandbits(32), "x3": self.rng.getrandbits(32),
            "x4": self.rng.getrandbits(48), "x5": self.rng.getrandbits(32), "flags": 0, "sp": 4096,
        }

    def _reif(self, x0_set):
        regs = dict(self.base)
        regs["x0"] = FAULTY_BASE + x0_set * 64
        spec = {"registers": regs, "memory": {"main_region": {}, "faulty_region": {}}}
        blob, _ = gri.generate_reif(spec, random.Random(self.seed))
        path = os.path.join(self.tmp, f"sym{x0_set}.reif")
        with open(path, "wb") as f:
            f.write(blob)
        return path

    def send(self, x0_set):
        """Load a fresh test case + the input whose [x0] maps to x0_set. Returns the input id."""
        _exe(CLEAR)
        _exe(CHECKOUT_TEST)
        _exe("w", TESTCASE)
        iid = [t for t in _exe(ALLOC).split() if t.isdigit()][-1]
        _exe(CHECKOUT_INPUT, iid)
        _exe("w", self._reif(x0_set))
        return iid

    def _trace(self, iid):
        _exe(TRACE)
        _exe(CHECKOUT_INPUT, iid)
        for line in _exe(MEASURE).splitlines():
            if "htrace 0" in line:
                bits = line.split()[-1]
                return {63 - i for i, c in enumerate(bits) if c == "1"}
        return set()

    def recover(self, iid, alphabet, traces):
        """Majority vote over `traces` traces: the alphabet set most often in L2 is the symbol."""
        votes = {c: 0 for c in alphabet}
        for _ in range(traces):
            hit = self._trace(iid)
            for c in alphabet:
                if c in hit:
                    votes[c] += 1
        return max(alphabet, key=lambda c: votes[c]), votes


def play_id(ch, candidates, rounds, traces, hard=False):
    print("=" * 68)
    print(f"  {'HARD ' if hard else ''}ID GAME -- recover the secret [x0] set from the trace")
    print(f"  candidates : {candidates}")
    print(f"  anchor set : {ch.anchor}   seed: {ch.seed}   {traces} traces/round")
    print("=" * 68)
    rng = random.SystemRandom()
    correct = 0
    for r in range(1, rounds + 1):
        secret = rng.choice(candidates)          # host's secret
        iid = ch.send(secret)
        guess, votes = ch.recover(iid, candidates, traces)   # sees only iid + public alphabet
        ok = guess == secret
        correct += ok
        vstr = " ".join(f"{c}:{votes[c]}" for c in candidates)
        print(f"  round {r:2d}: secret=set{secret:2d} | [{vstr}] -> guess=set{guess:2d}  "
              f"{'CORRECT' if ok else 'WRONG'}")
    print("=" * 68)
    print(f"  ACCURACY: {correct}/{rounds} = {correct / rounds:.0%}")


def play_message(ch, message, alphabet, traces):
    """A byte = high nibble then low nibble; each nibble n -> alphabet[n] (16-set alphabet)."""
    assert len(alphabet) == 16, "message mode needs a 16-set nibble alphabet"
    data = message.encode()
    symbols = [nib for b in data for nib in (b >> 4, b & 0xF)]   # secret plaintext -> nibble stream
    print("=" * 68)
    print("  MESSAGE MODE -- encode a string across the cache sets and recover it")
    print(f"  message  : {message!r} ({len(data)} bytes, {len(symbols)} nibble-symbols)")
    print(f"  alphabet : nibble n -> set {alphabet[0]}..{alphabet[-1]} (16 sets)   {traces} traces/symbol")
    print(f"  anchor   : set {ch.anchor}   seed: {ch.seed}")
    print("=" * 68)
    recovered_nibbles = []
    for i, nib in enumerate(symbols):
        iid = ch.send(alphabet[nib])                       # transmit the nibble as its set
        guess_set, _ = ch.recover(iid, alphabet, traces)   # receiver: vote among the 16 alphabet sets
        recovered_nibbles.append(alphabet.index(guess_set))
    rbytes = bytes((recovered_nibbles[2 * i] << 4) | recovered_nibbles[2 * i + 1]
                   for i in range(len(recovered_nibbles) // 2))
    recovered = rbytes.decode(errors="replace")
    good = sum(a == b for a, b in zip(data, rbytes))
    print(f"  sent      : {message!r}")
    print(f"  recovered : {recovered!r}")
    print("=" * 68)
    print(f"  bytes correct: {good}/{len(data)}   {'PERFECT RECOVERY' if recovered == message else ''}")


def main():
    ap = argparse.ArgumentParser(description="Cache side-channel game.")
    sub = ap.add_subparsers(dest="mode", required=True)

    p_id = sub.add_parser("id", help="recover one of N candidate [x0] sets")
    p_id.add_argument("--candidates", default="3,9,15,22,30,41,50,58")
    p_id.add_argument("--anchor", type=int, default=36)
    p_id.add_argument("--rounds", type=int, default=10)
    p_id.add_argument("--traces", type=int, default=5)
    p_id.add_argument("--seed", type=int, default=None)

    p_hard = sub.add_parser("hard", help="recover among ADJACENT sets (single cache line apart)")
    p_hard.add_argument("--base", type=int, default=20, help="first of the consecutive candidate sets")
    p_hard.add_argument("--span", type=int, default=6, help="how many consecutive sets")
    p_hard.add_argument("--anchor", type=int, default=40)
    p_hard.add_argument("--rounds", type=int, default=12)
    p_hard.add_argument("--traces", type=int, default=5)
    p_hard.add_argument("--seed", type=int, default=None)

    p_msg = sub.add_parser("message", help="encode + recover a text message across the sets")
    p_msg.add_argument("--message", default="REVIZOR N3")
    p_msg.add_argument("--alphabet", default=None, help="16 comma-separated sets for nibbles 0..15")
    p_msg.add_argument("--anchor", type=int, default=40)
    p_msg.add_argument("--traces", type=int, default=3)
    p_msg.add_argument("--seed", type=int, default=None)

    args = ap.parse_args()
    if not os.path.exists(DEVICE):
        sys.exit(f"[!] {DEVICE} not found -- load the kernel module first.")
    seed = args.seed if args.seed is not None else random.SystemRandom().randrange(1 << 30)
    configure_executor()
    ch = Channel(args.anchor, seed)

    if args.mode == "id":
        candidates = [int(x) for x in args.candidates.split(",")]
        if args.anchor in candidates:
            sys.exit("[!] --anchor must differ from every candidate.")
        play_id(ch, candidates, args.rounds, args.traces)
    elif args.mode == "hard":
        candidates = [args.base + i for i in range(args.span)]
        if args.anchor in candidates:
            sys.exit("[!] --anchor must not fall inside the candidate span.")
        play_id(ch, candidates, args.rounds, args.traces, hard=True)
    elif args.mode == "message":
        if args.alphabet:
            alphabet = [int(x) for x in args.alphabet.split(",")]
        else:
            alphabet = [2 + 2 * i for i in range(16)]   # sets 2,4,...,32
        if args.anchor in alphabet:
            sys.exit("[!] --anchor must not be in the alphabet.")
        play_message(ch, args.message, alphabet, args.traces)


if __name__ == "__main__":
    main()
