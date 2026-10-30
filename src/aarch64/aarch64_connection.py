"""
File: AArch64 remote-execution transports (SSH/ADB) for driving the executor on a remote device.
"""
import os
import shlex
import shutil
import socket
import struct
import subprocess

from ppadb.client import Client as AdbClient
import paramiko

from .aarch64_kernel import profile_op


class Connection:
    def __init__(self):
        pass

    def shell(self, command: str, privileged = False) -> str:
        raise NotImplementedError

    def run(self, command: str, stdin: bytes, privileged = False) -> bytes:
        """Run `command`, feeding `stdin` and returning stdout as raw bytes (for streamed batches)."""
        raise NotImplementedError

    def push(self, src, dst):
        raise NotImplementedError

    def pull(self, src, dst):
        raise NotImplementedError

    def is_file_present(self, filename: str) -> bool:
        raise NotImplementedError


class LocalConnection(Connection):
    """Drives the executor utilities on this machine via subprocess — no SSH. push/pull are copies."""

    def __init__(self):
        super().__init__()
        self._sudo = [] if 0 == os.geteuid() else ["sudo"]

    def _argv(self, command: str, privileged: bool):
        return (self._sudo if privileged else []) + ["bash", "-c", command]

    def shell(self, command: str, privileged = False) -> str:
        return self.run(command, b"", privileged).decode().strip()

    def run(self, command: str, stdin: bytes, privileged = False) -> bytes:
        p = subprocess.run(self._argv(command, privileged), input=stdin,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if 0 != p.returncode:
            raise IOError(f"local command failed (rc={p.returncode}): {command}\n"
                          f"{p.stderr.decode(errors='replace')}")
        return p.stdout

    def push(self, src, dst):
        if os.path.abspath(src) != os.path.abspath(dst):
            shutil.copy(src, dst)

    def pull(self, src, dst):
        if os.path.abspath(src) != os.path.abspath(dst):
            shutil.copy(src, dst)

    def is_file_present(self, filename: str) -> bool:
        return os.path.exists(filename)


class SSHConnection(Connection):
    def __init__(self, host: str = '127.0.0.1', port: int = 22, username: str = None, password: str = None, key_filename: str = None):
        super(SSHConnection, self).__init__()
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.client = paramiko.SSHClient()
        self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            self.client.connect(
                    hostname=self.host,
                    port=self.port,
                    username=self.username,
                    password=password,
                    key_filename=key_filename,
                    look_for_keys=True
                )
        except Exception as e:
            raise IOError(f'Could not connect to SSH server: {e}')

        try:
            self.sftp = self.client.open_sftp()
        except Exception as e:
            self.client.close()
            raise IOError(f'Could not open SFTP session: {e}')

    def shell(self, cmd: str, privileged = False) -> str:
        cmd = f'sudo -S bash -c {shlex.quote(cmd)}' if privileged else cmd
        stdin, stdout, stderr = self.client.exec_command(cmd)

        if privileged and stdin is not None:
            password = self.password if self.password else ""
            stdin.write(password + "\n")
            stdin.flush()

        out = stdout.read().decode().strip()
        err = stderr.read().decode().strip()
        rc = stdout.channel.recv_exit_status()
        if 0 != rc:
            raise IOError(f'remote command failed (rc={rc}): {cmd}\n{err}')
        return out

    def run(self, command: str, stdin: bytes, privileged = False) -> bytes:
        if privileged:
            command = f'sudo bash -c {shlex.quote(command)}'
        remote_in, remote_out, remote_err = self.client.exec_command(command)
        remote_in.write(stdin)
        remote_in.flush()
        remote_in.channel.shutdown_write()
        out = remote_out.read()
        rc = remote_out.channel.recv_exit_status()
        if 0 != rc:
            raise IOError(f'remote command failed (rc={rc}): {command}\n'
                          f'{remote_err.read().decode(errors="replace")}')
        return out

    def push(self, src, dst):
        with profile_op('push'):
            return self.sftp.put(src, dst)

    def pull(self, src, dst):
        with profile_op('pull'):
            return self.sftp.get(src, dst)

    def is_file_present(self, filename: str) -> bool:
        try:
            self.sftp.stat(filename)
            return True
        except FileNotFoundError:
            return False

    def close(self):
        if self.sftp:
            self.sftp.close()
        if self.client:
            self.client.close()


# adb shell-protocol v2 packet ids (platform_system_core shell_protocol.h): stdout, stderr and the
# exit code travel on separate framed streams, so a binary stdout is never polluted by a diagnostic
# the command writes to stderr (the plain `exec:` service merges the two).
_ADB_SHELL_STDIN = 0
_ADB_SHELL_STDOUT = 1
_ADB_SHELL_STDERR = 2
_ADB_SHELL_EXIT = 3
_ADB_SHELL_CLOSE_STDIN = 4
_ADB_SHELL_STDIN_CHUNK = 1 << 16


class ADBConnection(Connection):
    def __init__(self, host: str = '127.0.0.1', port: int = 5037, serial: str = None):
        super().__init__()
        self.host = host
        self.port = port
        self.client = AdbClient(host=self.host, port=self.port)

        if self.client is None:
            raise IOError('Could not connect to AdbClient')

        if len(self.client.devices()) == 0:
            raise IOError('Could not find devices for ADB')

        if serial is not None:
            self.device = next((d for d in self.client.devices() if d.serial == serial), None)
            if self.device is None:
                raise IOError(f'No ADB device with serial {serial!r}')
        else:
            self.device = self.client.devices()[0]
        self.serial = self.device.serial
        # The batch runs without su (su's PTY corrupts binary), so the shell domain must open the
        # device directly — which needs SELinux permissive. Relax it only when it is actually
        # enforcing, and verify it took rather than failing silently.
        if self.shell("getenforce", privileged=True).strip() == "Enforcing":
            self.shell("setenforce 0", privileged=True)
            mode = self.shell("getenforce", privileged=True).strip()
            if mode != "Permissive":
                raise IOError(f"the non-su batch needs SELinux permissive, but getenforce is {mode!r}")

    def shell(self, cmd: str, privileged = False) -> str:
        # adb shell does not surface the remote exit status, so append it and check it
        inner = f'su -c "{cmd}"' if privileged else cmd
        marker = "__RVZR_RC__"
        out = self.device.shell(f'{inner}; echo {marker}$?')
        idx = out.rfind(marker)
        tail = out[idx + len(marker):].split() if 0 <= idx else []
        if not tail:
            raise IOError(f'ADB shell: missing exit status for command {cmd!r}\n{out}')
        rc = int(tail[0])
        if 0 != rc:
            raise IOError(f'ADB shell command failed (rc={rc}): {cmd!r}\n{out[:idx]}')
        return out[:idx]

    def run(self, command: str, stdin: bytes, privileged = False) -> bytes:
        """Stream `stdin` to `command` over adb's shell-protocol v2 (raw, no PTY) and return its raw
        stdout. v2 frames stdout, stderr and the exit code on separate streams, so the binary batch
        response on stdout is never polluted by a diagnostic the command writes to stderr; a nonzero
        exit raises IOError carrying that stderr (matching the SSH/local transports)."""
        cmd = f"su -c '{command}'" if privileged else command
        s = socket.create_connection((self.host, self.port), timeout=600)
        s.settimeout(600)
        try:
            def recv_exact(n: int) -> bytes:
                buf = bytearray()
                while len(buf) < n:
                    chunk = s.recv(n - len(buf))
                    if not chunk:
                        raise IOError("adb connection closed mid-packet")
                    buf += chunk
                return bytes(buf)

            def service(request: str) -> None:
                s.sendall(f"{len(request):04x}{request}".encode())
                status = s.recv(4)
                if status != b"OKAY":
                    raise IOError(f"adb rejected {request!r}: {status!r} {s.recv(4096)!r}")

            def send_packet(packet_id: int, payload: bytes) -> None:
                s.sendall(bytes([packet_id]) + struct.pack("<I", len(payload)) + payload)

            service(f"host:transport:{self.serial}")
            service(f"shell,v2,raw:{cmd}")

            for off in range(0, len(stdin), _ADB_SHELL_STDIN_CHUNK):
                send_packet(_ADB_SHELL_STDIN, stdin[off:off + _ADB_SHELL_STDIN_CHUNK])
            send_packet(_ADB_SHELL_CLOSE_STDIN, b"")

            out, err, rc = bytearray(), bytearray(), None
            while rc is None:
                packet_id = recv_exact(1)[0]
                (length,) = struct.unpack("<I", recv_exact(4))
                payload = recv_exact(length)
                if _ADB_SHELL_STDOUT == packet_id:
                    out += payload
                elif _ADB_SHELL_STDERR == packet_id:
                    err += payload
                elif _ADB_SHELL_EXIT == packet_id:
                    rc = payload[0] if payload else -1
            if 0 != rc:
                raise IOError(f"remote command failed (rc={rc}): {cmd}\n"
                              f"{err.decode(errors='replace')}")
            return bytes(out)
        finally:
            s.close()

    def push(self, src, dst):
        return self.device.push(src, dst)

    def pull(self, src, dst):
        return self.device.pull(src, dst)

    def is_file_present(self, filename: str) -> bool:
        # shell() raises on a nonzero exit status, so `ls` of a missing file raises rather than
        # returning; treat that as absent instead of letting it propagate.
        try:
            self.shell(f'ls {filename}')
            return True
        except IOError:
            return False
