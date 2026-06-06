"""
Copyright (C) Microsoft Corporation
SPDX-License-Identifier: MIT
"""
import unittest

import copy
import os
import tempfile

from src.isa_loader import InstructionSet
from src.interfaces import OT, InstructionSpec
from src.config import CONF

basic = """
[
{"name": "test", "category": "CATEGORY", "control_flow": true,
  "operands": [
    {"type_": "MEM", "values": [], "src": true, "dest": true, "width": 16},
    {"type_": "REG", "values": ["ax"], "src": true, "dest": false, "width": 16}
  ],
  "implicit_operands": [
    {"type_": "FLAGS", "values": ["w", "r", "undef", "w", "w", "", "", "", "w"],
     "src": false, "dest": false, "width": 0}
  ]
}
]
"""

duplicate = """
[
{"name": "test", "category": "CATEGORY", "control_flow": false,
  "operands": [
    {"type_": "MEM", "values": [], "src": true, "dest": true, "width": 16}
  ],
  "implicit_operands": []
},
{"name": "test", "category": "CATEGORY", "control_flow": false,
  "operands": [
    {"type_": "MEM", "values": [], "src": true, "dest": true, "width": 16}
  ],
  "implicit_operands": []
}
]
"""


class InstructionSetParserTest(unittest.TestCase):

    def setUp(self):
        # CONF is a global (Borg) singleton, so another test's arch defaults can leak
        # in. Snapshot it, then select the generic (non-aarch64) load path with no
        # instruction filter — what these synthetic fixtures assume. tearDown restores.
        self._saved = copy.deepcopy(CONF._borg_shared_state)
        CONF.instruction_set = "x86-64"
        CONF.supported_instructions = None

    def tearDown(self):
        CONF._borg_shared_state.clear()
        CONF._borg_shared_state.update(self._saved)

    def test_parsing(self):
        spec_file = tempfile.NamedTemporaryFile("w", delete=False)
        with open(spec_file.name, "w") as f:
            f.write(basic)

        instruction_set = InstructionSet(spec_file.name)
        spec_file.close()
        os.unlink(spec_file.name)

        spec: InstructionSpec = instruction_set.instructions[0]
        self.assertEqual(spec.name, "test")
        self.assertEqual(spec.category, "CATEGORY")
        self.assertEqual(spec.has_mem_operand, True)
        self.assertEqual(spec.has_write, True)
        self.assertEqual(spec.control_flow, True)

        self.assertEqual(len(spec.operands), 2)
        op1 = spec.operands[0]
        self.assertEqual(op1.type, OT.MEM)
        self.assertEqual(op1.width, 16)
        self.assertEqual(op1.src, True)
        self.assertEqual(op1.dest, True)

        op2 = spec.operands[1]
        self.assertEqual(op2.type, OT.REG)
        self.assertEqual(op2.values, ["ax"])
        self.assertEqual(op2.src, True)
        self.assertEqual(op2.dest, False)

        self.assertEqual(len(spec.implicit_operands), 1)
        flags = spec.implicit_operands[0]
        self.assertEqual(flags.type, OT.FLAGS)
        self.assertEqual(flags.values, ['w', 'r', 'undef', 'w', 'w', '', '', '', 'w'])

    def test_dedup_identical(self):
        spec_file = tempfile.NamedTemporaryFile("w", delete=False)
        with open(spec_file.name, "w") as f:
            f.write(duplicate)

        instruction_set = InstructionSet(spec_file.name)
        spec_file.close()
        os.unlink(spec_file.name)

        self.assertEqual(len(instruction_set.instructions), 1, "No deduplication")


class CategoryFilterAnySemanticsTest(unittest.TestCase):
    """The category filter keeps an instruction if ANY of its tags is included."""

    def setUp(self):
        # CONF is a Borg singleton; snapshot and make the gates permissive so the
        # test isolates the tag filter (no supported/blocklist/register gating).
        self._saved = copy.deepcopy(CONF._borg_shared_state)
        CONF.instruction_set = "x86-64"   # avoid the aarch64 "general"-only gate
        CONF.supported_instructions = []
        CONF.instruction_blocklist = []
        CONF.instruction_allowlist = []
        CONF.register_blocklist = []
        CONF.register_allowlist = []
        CONF._no_generation = False

    def tearDown(self):
        CONF._borg_shared_state.clear()
        CONF._borg_shared_state.update(self._saved)

    @staticmethod
    def _reduce(specs, include):
        iset = InstructionSet.__new__(InstructionSet)
        iset.instructions = list(specs)
        iset.reduce(include)
        return {i.name for i in iset.instructions}

    def test_kept_when_any_tag_matches(self):
        # tags {A, B}, include {A} -> kept under any() (would be dropped under all())
        spec = InstructionSpec("multi", "general", tags=["TAG-A", "TAG-B"])
        self.assertIn("multi", self._reduce([spec], ["TAG-A"]))

    def test_dropped_when_no_tag_matches(self):
        spec = InstructionSpec("none", "general", tags=["TAG-A", "TAG-B"])
        self.assertNotIn("none", self._reduce([spec], ["TAG-C"]))

    def test_mixed_set(self):
        hit = InstructionSpec("hit", "general", tags=["X", "Y"])
        miss = InstructionSpec("miss", "general", tags=["Z"])
        self.assertEqual(self._reduce([hit, miss], ["X"]), {"hit"})
