"""
Copyright (C) Microsoft Corporation
SPDX-License-Identifier: MIT
"""
import unittest
import inspect
import pathlib
import copy
from src.config import CONF

FILE_DIR = pathlib.Path(__file__).parent.resolve()
DOC_DIR = FILE_DIR.parent / "docs"


class DocumentationTest(unittest.TestCase):
    """
    A class for testing if the documentation is up to date.
    """
    longMessage = False

    def setUp(self):
        # The common config docs are x86-oriented; apply the arch defaults so CONF carries the
        # merged _option_values/defaults the doc describes. (CONF no longer self-initializes an
        # arch at import; arch defaults are applied via set_to_arch_defaults at config load.)
        self._saved = copy.deepcopy(CONF._borg_shared_state)
        CONF.instruction_set = "x86-64"
        CONF.set_to_arch_defaults()

    def tearDown(self):
        CONF._borg_shared_state.clear()
        CONF._borg_shared_state.update(self._saved)

    def test_conf_docs(self):
        """
        Test that every config option is documented. Common options live in the common config
        reference (docs/user/config.md); architecture-specific options live in their architecture's
        config reference (docs/<arch>/config.md). We accept the option in any config reference.
        """
        docs = "".join(p.read_text() for p in sorted(DOC_DIR.glob("*/config.md")))

        options = [
            k[0]
            for k in inspect.getmembers(CONF, lambda x: not inspect.isroutine(x))
            if not k[0].startswith("_")
        ]

        for option in options:
            self.assertTrue(option in docs, msg=f"{option} not found in documentation")

    def test_conf_options_docs(self):
        """
        Test if the documentation contains all possible values for the config options.
        """
        # get the text of the config documentation
        with open(DOC_DIR / "user/config.md", "r") as f:
            doc_text = f.readlines()

        # build a map of config options to their possible values in the doc
        doc_options = {}
        curr_name = ""
        for line in doc_text:
            if line.startswith("Name"):
                curr_name = line.split(":")[1].strip()
                doc_options[curr_name] = ["", []]
            elif line.startswith("Actor Option:"):
                curr_name = f"actor_{line.split(':')[1].strip()}"
                doc_options[curr_name] = ["", []]
            elif line.startswith("Default:"):
                doc_options[curr_name][0] = line.split(":")[1].strip().strip("'")
            elif line.startswith("Options:"):
                if "(" in line:
                    continue
                doc_options[curr_name][1] = line.split(":")[1].strip().split("|")
                for i in range(len(doc_options[curr_name][1])):
                    doc_options[curr_name][1][i] = doc_options[curr_name][1][i].strip().strip("'")

        # get a list of config options
        options = [
            k for k in inspect.getmembers(CONF, lambda x: not inspect.isroutine(x))
            if not k[0].startswith("_")
        ]
        alternatives = CONF._option_values

        # check if all alternatives and defaults are documented
        for name, default_ in options:
            if name not in doc_options:
                continue
            if not doc_options[name][0].startswith("("):
                self.assertEqual(
                    str(default_),
                    doc_options[name][0],
                    msg=f"Default for `{name}` is incorrect: {default_} != {doc_options[name][0]}"
                )

            if doc_options[name][1]:
                doc_values = doc_options[name][1]
                self.assertSetEqual(
                    set(alternatives[name]),
                    set(doc_values),
                    msg=f"Options for `{name}` are incorrect: {alternatives[name]} != {doc_values}"
                )
