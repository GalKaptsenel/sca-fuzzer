import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))  # run from any cwd
import unittest
from src.aarch64.aarch64_generator import Aarch64Generator


class InMemoryAssembleCacheTest(unittest.TestCase):
    """in_memory_assemble() is a pure function of the (normalized) source text and
    is memoized to avoid the process spawns (asm_to_bytes -> as + objcopy) on the
    sealed/NI path, where the same placeholder test case is re-assembled per input."""

    def test_encoding_is_correct(self):
        # NOP encodes to 0xd503201f (little-endian bytes 1f 20 03 d5).
        self.assertEqual(Aarch64Generator.in_memory_assemble("nop"), b"\x1f\x20\x03\xd5")

    def test_repeat_is_a_cache_hit_not_a_respawn(self):
        Aarch64Generator._assemble_cached.cache_clear()
        first = Aarch64Generator.in_memory_assemble("mov x0, #1\n")
        after_first = Aarch64Generator._assemble_cached.cache_info()
        self.assertEqual(after_first.misses, 1)
        self.assertEqual(after_first.hits, 0)

        second = Aarch64Generator.in_memory_assemble("mov x0, #1\n")
        after_second = Aarch64Generator._assemble_cached.cache_info()
        self.assertEqual(second, first)              # identical bytes
        self.assertEqual(after_second.misses, 1)     # not re-assembled
        self.assertEqual(after_second.hits, 1)       # served from cache

    def test_trailing_newline_normalized_to_same_entry(self):
        Aarch64Generator._assemble_cached.cache_clear()
        a = Aarch64Generator.in_memory_assemble("nop")     # gets a '\n' appended
        b = Aarch64Generator.in_memory_assemble("nop\n")   # already normalized
        self.assertEqual(a, b)
        self.assertEqual(Aarch64Generator._assemble_cached.cache_info().misses, 1)


if __name__ == "__main__":
    unittest.main()
