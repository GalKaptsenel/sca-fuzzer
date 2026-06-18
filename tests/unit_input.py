"""Input.__new__ zero-initializes the whole buffer (incl. the padding field) so tobytes()/__hash__
are deterministic -- two equal-content inputs must hash the same."""
import unittest
import numpy as np

from src.interfaces import Input


class InputHashTest(unittest.TestCase):
    def test_fresh_input_is_zeroed(self):
        for _ in range(20):
            if Input().view(np.uint64).any():
                self.fail("fresh Input is not zero-initialized (uninitialized buffer/padding)")

    def test_equal_content_inputs_hash_equal(self):
        a = Input()
        b = Input()
        a['main'][0][0] = 0xDEAD
        b['main'][0][0] = 0xDEAD
        self.assertEqual(hash(a), hash(b))


if __name__ == "__main__":
    unittest.main()
