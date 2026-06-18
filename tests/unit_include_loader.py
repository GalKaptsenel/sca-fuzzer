"""
Copyright (C) Microsoft Corporation
SPDX-License-Identifier: MIT
"""
import os
import tempfile
import unittest

from src.config import IncludeLoader, ConfigException


def _load(path: str):
    with open(path, "r") as f:
        loader = IncludeLoader(f)
        try:
            return loader.get_single_data()
        finally:
            loader.dispose()


def _write(path: str, text: str) -> None:
    with open(path, "w") as f:
        f.write(text)


class IncludeLoaderTest(unittest.TestCase):
    """ Regression: include-cycle tracking must be per-load instance state, not GC-timed
    class state (which leaks across loads). """

    def test_no_class_level_visited_state(self):
        self.assertFalse(
            hasattr(IncludeLoader, "visited"),
            "IncludeLoader.visited class state reintroduced (cross-load pollution risk)")

    def test_circular_include_detected(self):
        with tempfile.TemporaryDirectory() as d:
            _write(os.path.join(d, "a.yml"), "a: !include b.yml\n")
            _write(os.path.join(d, "b.yml"), "b: !include a.yml\n")
            with self.assertRaises(ConfigException):
                _load(os.path.join(d, "a.yml"))

    def test_normal_include_loads(self):
        with tempfile.TemporaryDirectory() as d:
            _write(os.path.join(d, "a.yml"), "a: !include b.yml\n")
            _write(os.path.join(d, "b.yml"), "value: 42\n")
            self.assertEqual(_load(os.path.join(d, "a.yml")), {"a": {"value": 42}})


if __name__ == "__main__":
    unittest.main()
