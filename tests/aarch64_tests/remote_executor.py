import random
import numpy as np
import unittest
from unittest.mock import Mock, call
from src.aarch64.aarch64_executor import Aarch64RemoteExecutor
from src.interfaces import InputFragment, Input

class TestRemoteExecutor(unittest.TestCase):

    def setUp(self):
        input_counter = 0

        def custom_shell(cmd: str) -> str:
            nonlocal input_counter
            split_cmd = cmd.split()
            if len(split_cmd) >= 2 and split_cmd[0] == "ls":
                return split_cmd[1]
            elif cmd == 'cat /sys/devices/system/cpu/smt/control':
                return 'notimplemented'
            elif len(split_cmd) >= 2 and split_cmd[0] == "su" and split_cmd[1] == "-c":
                if 'cat' in cmd:
                    if '/sys/executor/print_code_base' in cmd:
                        return '000000008bf3e4a0'
                    if '/sys/executor/print_sandbox_base' in cmd:
                        return '00000000bbb328c7'
                if '/data/local/tmp/executor_userland' in cmd:
                    if len(split_cmd) < 5:
                        raise ValueError
                    assert split_cmd[3] == '/dev/executor'
                    if split_cmd[4] == 'w':
                        return ''
                    service_nr = int(split_cmd[4].replace("'", ""). replace('"', ""). replace(" ", ""))
                    if service_nr == 1:
                        return """
                        Device: /dev/executor
                        Command: 29185
                        Command magic: 114
                        Command number: 1
                        IOCTL command executed successfully.
                        Result: 0x0
                        """
                    elif service_nr == 4:
                        if len(split_cmd) < 6:
                            raise ValueError
                        iid = int(split_cmd[5].replace("'", ""). replace('"', ""). replace(" ", ""))
                        return f"""
                        Device: /dev/executor
                        Command: 1074295300
                        Command magic: 114
                        Command number: 4
                        Argument: 0x{iid}
                        IOCTL command executed successfully.
                        Result: 0x0
                        """
                    elif service_nr == 5:
                        returned_number = input_counter
                        input_counter += 1
                        return f"""
                        Device: /dev/executor
                        Command: -2146930171
                        Command magic: 114
                        Command number: 5
                        Allocated Input ID: {returned_number}
                        IOCTL command executed successfully.
                        Result: 0x0
                        """
                    elif service_nr == 8:
                        return """
                        Device: /dev/executor
                        Command: 29192
                        Command magic: 114
                        Command number: 8
                        IOCTL command executed successfully.
                        Result: 0x0
                        """
                    elif service_nr == 7:
                        random_pfc1 = random.randint(0, 64)
                        random_pfc2 = random.randint(0, 64)
                        random_pfc3 = random.randint(0, 64)
                        htrace_list = ['1'] * random_pfc3 + ['0'] * (64 - random_pfc3)
                        random.shuffle(htrace_list)
                        htrace = ''.join(htrace_list)
                        return f"""
                        Device: /dev/executor
                        Command: -2145357305
                        Command magic: 114
                        Command number: 7
                        Measurement:
                                htrace 1: {htrace}

                                pfc 1: {random_pfc1}
                                pfc 2: {random_pfc2}
                                pfc 3: {random_pfc3}
                        IOCTL command executed successfully.
                        Result: 0x0
                        """
                    else:
                        raise ValueError("Unexpected service number from executor_userland")

        self.mock_connection = Mock()
        self.mock_connection.shell.side_effect = custom_shell

    def test_remote_executor_setup(self):
        rexecutor = Aarch64RemoteExecutor(self.mock_connection, False)

        self.mock_connection.shell.assert_any_call('ls /dev/executor')
        self.mock_connection.shell.assert_any_call('ls /data/local/tmp/executor_userland')

    def test_remote_executor_read_base_addresses(self):
        rexecutor = Aarch64RemoteExecutor(self.mock_connection, False)

        result = rexecutor.read_base_addresses()
        assert result == (0xbbb328c7, 0x8bf3e4a0)

    def test_remote_executor__is_smt_enabled(self):
        rexecutor = Aarch64RemoteExecutor(self.mock_connection, False)

        result = result = rexecutor._is_smt_enabled()
        assert result is False
        self.mock_connection.shell.assert_any_call("cat /sys/devices/system/cpu/smt/control")

    def test_remote_executor__write_test_case(self):
        rexecutor = Aarch64RemoteExecutor(self.mock_connection, False)
        mock_testcase = Mock()
        mock_testcase.bin_path = 'generated.asm'
        remote_fname = "/data/local/tmp/generated.asm"

        rexecutor._write_test_case(mock_testcase)

        expected_calls = [call('su -c "/data/local/tmp/executor_userland /dev/executor 1"',), call(f'su -c "/data/local/tmp/executor_userland /dev/executor w {remote_fname}"',)]
        self.mock_connection.shell.assert_has_calls(expected_calls, any_order=False)
        self.mock_connection.push.assert_called_once_with(mock_testcase.bin_path, remote_fname)


    def test_remote_executor_trace_test_case(self):
        rexecutor = Aarch64RemoteExecutor(self.mock_connection, False)
        n_reps = 5
        inputs_amount = 2
        state = np.random.randint(0, 1000000)
        inputs = [Input() for i in range(inputs_amount)]

        size = inputs[0].itemsize // 8
        rng = np.random.default_rng(seed=state)

        for inp in inputs:
            for i in range(len(inp)): 

                data = rng.integers(2**64, size=size, dtype=np.uint64)

                inp[i] = data.view(InputFragment)
                inp[i]['padding'] = 0

        actual_inputs_flow = [i for i in range(inputs_amount)] * n_reps
        
        expected_call_chunks  = []

        # Assuming increasing order of input ids and the order of filenames used
        for iid, input_file_number in enumerate(actual_inputs_flow):
            filename = f'/data/local/tmp/input{input_file_number}.bin'
            expected_call_chunks.append([
                call('su -c "/data/local/tmp/executor_userland /dev/executor 5"'),
                ])
            expected_call_chunks.append([
                call(f'su -c "/data/local/tmp/executor_userland /dev/executor 4 {iid}"'),
                call(f'su -c "/data/local/tmp/executor_userland /dev/executor w {filename}"'),
                ])
            expected_call_chunks.append([
                call(f'su -c "/data/local/tmp/executor_userland /dev/executor 4 {iid}"'),
                call('su -c "/data/local/tmp/executor_userland /dev/executor 7"'),
                ])
            expected_call_chunks.append([
                call(f'su -c "rm {filename}"'),
                ])



        expected_call_chunks.append([call('su -c "/data/local/tmp/executor_userland /dev/executor 8"')])

        rexecutor.trace_test_case(inputs, n_reps) 

        for chunk in expected_call_chunks:
            self.mock_connection.shell.assert_has_calls(chunk, any_order=False)

