"""The template-regex cache is keyed by the spec object (WeakKeyDictionary), so a spec's entry is
evicted once the spec is collected -- no stale regex if its id() is later reused by a new spec."""
import gc
import unittest

from src.interfaces import InstructionSpec
import src.aarch64.aarch64_asm_parser as ap


class TemplateRegexCacheTest(unittest.TestCase):
    def test_dead_spec_is_evicted(self):
        ap._TEMPLATE_REGEX_CACHE.clear()
        spec = InstructionSpec(name="nop", template="nop")
        self.assertIsNotNone(ap._template_regex(spec))
        self.assertEqual(len(ap._TEMPLATE_REGEX_CACHE), 1)
        del spec
        gc.collect()
        self.assertEqual(len(ap._TEMPLATE_REGEX_CACHE), 0)

    def test_distinct_specs_get_distinct_regex(self):
        a = InstructionSpec(name="nop", template="nop")
        b = InstructionSpec(name="ret", template="ret")
        self.assertNotEqual(ap._template_regex(a).pattern, ap._template_regex(b).pattern)


if __name__ == "__main__":
    unittest.main()
