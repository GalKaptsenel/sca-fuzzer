"""Module-level isolation of the shared CONF (Borg) state and the working directory: a test module that
does `from tests.conf_isolation import setUpModule, tearDownModule` restores both when it finishes, so no
module's outcome depends on which modules ran before it."""
import copy
import os

from src.config import CONF

_saved = []


def setUpModule():
    _saved.append((copy.deepcopy(CONF._borg_shared_state), os.getcwd()))


def tearDownModule():
    conf, cwd = _saved.pop()
    CONF._borg_shared_state.clear()
    CONF._borg_shared_state.update(conf)
    os.chdir(cwd)
