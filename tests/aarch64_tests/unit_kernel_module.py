"""
AArch64 executor kernel-module tests (hardware-gated).

Two groups, both skipped unless the module is loaded:

  TestExecutorSysfs  - the /sys/executor sysfs interface: store-handler input
                       validation, read-only show handlers, the legacy combined
                       flush knob, branch_training_config round-trip.

  TestExecutorIoctl  - the /dev/executor ioctl state machine ("automata"):
                       allocate/free/clear bookkeeping, checkout, the
                       precondition errors (GET_TEST_LENGTH / TRACE), corner
                       cases (invalid ids, wrong magic), and one guarded
                       full lifecycle (load NOP test case -> trace -> measure).

Both groups snapshot and restore the device state they touch so they do not
leak configuration across tests or into a later fuzzing run.
"""
import os
import fcntl
import struct
import unittest

SYSFS = "/sys/executor"
DEVICE = "/dev/executor"
MODULE_LOADED = os.path.isdir(SYSFS)
DEVICE_PRESENT = os.path.exists(DEVICE)


def _measurement_supported() -> bool:
    try:
        with open(f"{SYSFS}/system/measurement_supported") as f:
            return "1" == f.read().strip()
    except OSError:
        return False


MEASUREMENT_SUPPORTED = _measurement_supported()


# =============================================================================
# sysfs
# =============================================================================
_BOOL_KNOBS = ["enable_phr_flush", "enable_view_rotation",
               "enable_ssbs", "enable_branch_training"]


def _spath(name):
    return os.path.join(SYSFS, name)


def _read(name):
    with open(_spath(name)) as f:
        return f.read()


def _write_raw(name, value):
    """Write via a direct syscall so a store-handler -EINVAL surfaces here, now."""
    fd = os.open(_spath(name), os.O_WRONLY)
    try:
        os.write(fd, value.encode() if isinstance(value, str) else value)
    finally:
        os.close(fd)


@unittest.skipUnless(MODULE_LOADED, "executor kernel module not loaded (/sys/executor absent)")
class TestExecutorSysfs(unittest.TestCase):

    def setUp(self):
        self._saved = {}
        snap = (["warmups", "measurement_mode", "enable_pre_run_flush",
                 "branch_training_config"] + _BOOL_KNOBS)
        for k in snap:
            try:
                self._saved[k] = _read(k).strip()
            except OSError:
                pass

    def tearDown(self):
        def restore(k):
            if k in self._saved:
                _write_raw(k, self._saved[k])
        restore("warmups")
        if "measurement_mode" in self._saved:
            _write_raw("measurement_mode",
                       "F+R" if "F+R" in self._saved["measurement_mode"] else "P+P")
        restore("enable_pre_run_flush")
        restore("enable_phr_flush")
        restore("enable_view_rotation")
        restore("enable_ssbs")
        # Restore the training config (empty -> "\n" clears it to 0 entries) BEFORE the
        # enable flag, since writing the config re-enables training as a side effect.
        if "branch_training_config" in self._saved:
            _write_raw("branch_training_config", self._saved["branch_training_config"] or "\n")
        restore("enable_branch_training")

    # ---- warmups: kstrtol, reject negative / non-numeric --------------------
    def test_warmups_accepts_nonneg_int(self):
        _write_raw("warmups", "7")
        self.assertEqual(int(_read("warmups").strip()), 7)

    def test_warmups_rejects_negative(self):
        with self.assertRaises(OSError):
            _write_raw("warmups", "-1")

    def test_warmups_rejects_garbage(self):
        with self.assertRaises(OSError):
            _write_raw("warmups", "abc")

    # ---- boolean knobs: kstrtobool, accept 0/1, reject garbage --------------
    def test_bool_knobs_accept_0_and_1(self):
        for k in _BOOL_KNOBS:
            _write_raw(k, "1")
            self.assertEqual(_read(k).strip(), "1", k)
            _write_raw(k, "0")
            self.assertEqual(_read(k).strip(), "0", k)

    def test_bool_knobs_reject_garbage(self):
        # kstrtobool keys off the first char (y/n/0/1/o), so use one it cannot parse.
        for k in _BOOL_KNOBS:
            with self.assertRaises(OSError, msg=k):
                _write_raw(k, "xyz")

    # ---- measurement_mode: only P+P / F+R -----------------------------------
    def test_measurement_mode_accepts_pp_fr(self):
        _write_raw("measurement_mode", "P+P")
        self.assertIn("P+P", _read("measurement_mode"))
        _write_raw("measurement_mode", "F+R")
        self.assertIn("F+R", _read("measurement_mode"))

    def test_measurement_mode_rejects_unknown(self):
        with self.assertRaises(OSError):
            _write_raw("measurement_mode", "X+Y")

    # ---- legacy combined flush knob drives the split knobs ------------------
    def test_pre_run_flush_drives_phr_and_rotation(self):
        _write_raw("enable_pre_run_flush", "1")
        self.assertEqual(_read("enable_phr_flush").strip(), "1")
        self.assertEqual(_read("enable_view_rotation").strip(), "1")
        _write_raw("enable_pre_run_flush", "0")
        self.assertEqual(_read("enable_phr_flush").strip(), "0")
        self.assertEqual(_read("enable_view_rotation").strip(), "0")

    # ---- read-only show handlers --------------------------------------------
    def test_print_bases_are_hex_pointers(self):
        # show handlers use %px -> bare hex, no "0x" prefix
        for k in ("print_sandbox_base", "print_code_base"):
            self.assertGreater(int(_read(k).strip(), 16), 0, k)

    def test_print_base_is_read_only(self):
        with self.assertRaises(OSError):
            _write_raw("print_sandbox_base", "0")

    # ---- branch_training_config round-trips + auto-enables training ---------
    def test_branch_training_config_roundtrip(self):
        _write_raw("branch_training_config", "8:1")
        self.assertIn("8:1", _read("branch_training_config"))
        self.assertEqual(_read("enable_branch_training").strip(), "1")


# =============================================================================
# ioctl automata
#
# _IOC encoding (asm-generic, used by arm64): dir<<30 | size<<16 | type<<8 | nr.
# Mirror of userapi/executor_ioctl_nr.h (magic 'r', consecutive NRs).
# =============================================================================
_IOC_NONE, _IOC_WRITE, _IOC_READ = 0, 1, 2


def _IOC(d, nr, size):
    return (d << 30) | (ord('r') << 8) | (nr << 0) | (size << 16)


CHECKOUT_TEST = _IOC(_IOC_NONE, 1, 0)
UNLOAD_TEST = _IOC(_IOC_NONE, 2, 0)
GET_NUM_INPUTS = _IOC(_IOC_READ, 3, 8)
CHECKOUT_INPUT = _IOC(_IOC_WRITE, 4, 8)
ALLOCATE_INPUT = _IOC(_IOC_READ, 5, 8)
FREE_INPUT = _IOC(_IOC_WRITE, 6, 8)
MEASUREMENT = _IOC(_IOC_READ, 7, 32)
TRACE = _IOC(_IOC_NONE, 8, 0)
CLEAR_ALL_INPUTS = _IOC(_IOC_NONE, 9, 0)
GET_TEST_LENGTH = _IOC(_IOC_READ, 10, 8)
# struct mte_tag_region_req { u64 sandbox_offset; u64 length; u8 tag; } -> 24 bytes (padded).
MTE_TAG_REGION = _IOC(_IOC_WRITE, 16, 24)
# Taggable span = lower_overflow|main|faulty|upper_overflow = 4 * sandbox PAGESIZE (4096 each).
MTE_TAGGABLE_SPAN = 4 * 4096

# main(4096) + faulty(4096) + 8 register slots * 8 bytes
USER_CONTROLLED_INPUT_LENGTH = 4096 + 4096 + 8 * 8
NOP = b"\x1f\x20\x03\xd5"   # aarch64 NOP, little-endian


@unittest.skipUnless(DEVICE_PRESENT, "executor device /dev/executor absent")
class TestExecutorIoctl(unittest.TestCase):

    def setUp(self):
        self.fd = os.open(DEVICE, os.O_RDWR)
        # The lifecycle test writes measurement_mode via sysfs; snapshot it so we
        # never leak a config change out of the suite.
        self._saved_mode = None
        if MODULE_LOADED:
            mode = _read("measurement_mode")
            if "F+R" in mode:
                self._saved_mode = "F+R"
            elif "P+P" in mode:
                self._saved_mode = "P+P"
        self._clear()  # start from a known CONFIGURATION state

    def tearDown(self):
        try:
            self._ioctl_void(UNLOAD_TEST)
            self._clear()
            if self._saved_mode is not None:
                _write_raw("measurement_mode", self._saved_mode)
        finally:
            os.close(self.fd)

    # ---- ioctl helpers ------------------------------------------------------
    def _ioctl_void(self, cmd):
        return fcntl.ioctl(self.fd, cmd, 0)

    def _ioctl_get_u64(self, cmd):
        buf = bytearray(8)
        fcntl.ioctl(self.fd, cmd, buf, True)
        return struct.unpack("<q", buf)[0]

    def _ioctl_put_u64(self, cmd, value):
        return fcntl.ioctl(self.fd, cmd, struct.pack("<q", value), False)

    def _clear(self):
        self._ioctl_void(CLEAR_ALL_INPUTS)

    def _num_inputs(self):
        return self._ioctl_get_u64(GET_NUM_INPUTS)

    def _tag_region(self, offset, length, tag=6):
        req = struct.pack("<QQB", offset, length, tag).ljust(24, b"\x00")
        return fcntl.ioctl(self.fd, MTE_TAG_REGION, req, False)

    # ---- MTE tag-region ioctl bounds (span = lower_overflow|main|faulty|upper_overflow) ------
    def test_mte_tag_region_accepts_full_span(self):
        self.assertEqual(self._tag_region(0, MTE_TAGGABLE_SPAN), 0)
        self.assertEqual(self._tag_region(3 * 4096, 4096), 0)   # the upper_overflow region

    def test_mte_tag_region_rejects_out_of_range(self):
        for offset, length in ((0, MTE_TAGGABLE_SPAN + 1),
                               (MTE_TAGGABLE_SPAN, 1),
                               (MTE_TAGGABLE_SPAN - 1, 2)):
            with self.assertRaises(OSError):
                self._tag_region(offset, length)

    def test_short_input_write_is_rejected(self):
        a = self._ioctl_get_u64(ALLOCATE_INPUT)
        self._ioctl_put_u64(CHECKOUT_INPUT, a)
        try:
            with self.assertRaises(OSError):                          # -EINVAL, not silent success
                os.write(self.fd, b"\x00" * (USER_CONTROLLED_INPUT_LENGTH - 1))
        finally:
            self._ioctl_put_u64(FREE_INPUT, a)

    # ---- input bookkeeping automata -----------------------------------------
    def test_clear_gives_zero_inputs(self):
        self.assertEqual(self._num_inputs(), 0)

    def test_allocate_increments_count(self):
        self.assertEqual(self._num_inputs(), 0)
        self._ioctl_get_u64(ALLOCATE_INPUT)
        self.assertEqual(self._num_inputs(), 1)
        self._ioctl_get_u64(ALLOCATE_INPUT)
        self.assertEqual(self._num_inputs(), 2)

    def test_allocated_id_is_usable(self):
        # No assumption about the id's value/order: whatever ALLOCATE returns
        # must be checkout-able and free-able.
        a = self._ioctl_get_u64(ALLOCATE_INPUT)
        self._ioctl_put_u64(CHECKOUT_INPUT, a)     # must not error
        self._ioctl_put_u64(FREE_INPUT, a)
        self.assertEqual(self._num_inputs(), 0)

    def test_free_decrements_count(self):
        a = self._ioctl_get_u64(ALLOCATE_INPUT)
        self._ioctl_get_u64(ALLOCATE_INPUT)
        self.assertEqual(self._num_inputs(), 2)
        self._ioctl_put_u64(FREE_INPUT, a)
        self.assertEqual(self._num_inputs(), 1)

    # ---- corner cases: invalid ids are soft no-ops (never crash) ------------
    def test_free_negative_id_is_noop(self):
        self._ioctl_get_u64(ALLOCATE_INPUT)
        self._ioctl_put_u64(FREE_INPUT, -1)        # negative is invalid by definition
        self.assertEqual(self._num_inputs(), 1)

    def test_double_free_is_noop(self):
        # The second free targets an id that no longer exists (assumption-free way
        # to exercise "free a non-existent id").
        a = self._ioctl_get_u64(ALLOCATE_INPUT)
        self._ioctl_put_u64(FREE_INPUT, a)
        self._ioctl_put_u64(FREE_INPUT, a)
        self.assertEqual(self._num_inputs(), 0)

    def test_checkout_freed_input_is_safe(self):
        # Checking out an id that was freed (now non-existent) must be a safe no-op.
        a = self._ioctl_get_u64(ALLOCATE_INPUT)
        self._ioctl_put_u64(FREE_INPUT, a)
        self._ioctl_put_u64(CHECKOUT_INPUT, a)

    # ---- precondition automata (these DO return errno) ----------------------
    def test_get_test_length_without_test_fails(self):
        # CONFIGURATION state (no test loaded) -> -EINVAL
        with self.assertRaises(OSError):
            self._ioctl_get_u64(GET_TEST_LENGTH)

    def test_trace_without_ready_fails(self):
        # no test + no inputs -> not READY/TRACED -> -EINVAL
        with self.assertRaises(OSError):
            self._ioctl_void(TRACE)

    def test_measurement_without_trace_fails(self):
        # not TRACED -> measure returns -EINVAL (must reach userspace, not be swallowed)
        with self.assertRaises(OSError):
            fcntl.ioctl(self.fd, MEASUREMENT, bytearray(32), True)

    def test_unknown_magic_ioctl_rejected(self):
        bad = (_IOC_NONE << 30) | (ord('Z') << 8) | 1   # valid shape, wrong magic -> -ENOTTY
        with self.assertRaises(OSError):
            fcntl.ioctl(self.fd, bad, 0)

    # ---- test-case load automata --------------------------------------------
    def test_load_test_sets_then_clears_length(self):
        self._ioctl_void(CHECKOUT_TEST)
        os.write(self.fd, NOP * 4)
        self.assertEqual(self._ioctl_get_u64(GET_TEST_LENGTH), len(NOP) * 4)
        self._ioctl_void(UNLOAD_TEST)
        with self.assertRaises(OSError):
            self._ioctl_get_u64(GET_TEST_LENGTH)

    # ---- guarded full lifecycle: load NOP test case -> trace -> measure -----
    @unittest.skipUnless(MEASUREMENT_SUPPORTED, "PMU measurement unsupported (need >=4 event counters)")
    def test_full_lifecycle_trace_and_measure(self):
        iid = self._ioctl_get_u64(ALLOCATE_INPUT)
        self._ioctl_put_u64(CHECKOUT_INPUT, iid)
        os.write(self.fd, bytes(USER_CONTROLLED_INPUT_LENGTH))   # all-zero input

        self._ioctl_void(CHECKOUT_TEST)
        os.write(self.fd, NOP * 8)                               # trivial, safe TC

        _write_raw("measurement_mode", "P+P")                    # template must be set before TRACE
        self._ioctl_void(TRACE)                                  # executes on HW; raises on error

        self._ioctl_put_u64(CHECKOUT_INPUT, iid)
        buf = bytearray(32)
        fcntl.ioctl(self.fd, MEASUREMENT, buf, True)             # htrace[1] + pfc[3]
        # measurement returned 32 bytes; just assert we got the structure back
        self.assertEqual(len(buf), 32)


@unittest.skipUnless(MODULE_LOADED, "executor kernel module not loaded (/sys/executor absent)")
class TestSystemSysfs(unittest.TestCase):
    """system/ subdirectory exposing host capability info."""

    def test_attributes_present_and_parse(self):
        self.assertGreaterEqual(int(_read("system/pmu_event_counters")), 0)
        self.assertIn(_read("system/measurement_supported").strip(), ("0", "1"))
        self.assertIn("MIDR_EL1", _read("system/cpu_info"))

    def test_measurement_supported_tracks_counter_count(self):
        counters = int(_read("system/pmu_event_counters"))
        supported = "1" == _read("system/measurement_supported").strip()
        self.assertEqual(supported, counters >= 4)   # config_pfc programs counters 0..3


if __name__ == "__main__":
    unittest.main(verbosity=2)
