"""stable_rng draws the same stream in every process, unlike random.Random(hash(...)) whose str/bytes
hashing is randomized per process (PYTHONHASHSEED) — which made decoy choices unreproducible."""
import os
import subprocess
import sys
import unittest

_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
_SNIPPET = ("import sys; sys.path.insert(0, {root!r}); from src.util import stable_rng; "
            "print(stable_rng(((5, False), (None, True)), 0x99, 'forced-noncanon').getrandbits(64), "
            "hash('forced-noncanon'))")


def _draw(hash_seed: str):
    env = dict(os.environ, PYTHONHASHSEED=hash_seed)
    out = subprocess.run([sys.executable, "-c", _SNIPPET.format(root=_ROOT)], env=env, check=True,
                         capture_output=True, text=True).stdout.split()
    return int(out[0]), int(out[1])


class StableRngTest(unittest.TestCase):
    def test_same_stream_across_processes(self):
        (a, ha), (b, hb) = _draw("1"), _draw("2")
        self.assertNotEqual(ha, hb, "control: str hash() must differ across hash seeds")
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
