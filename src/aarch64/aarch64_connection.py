"""
File: AArch64 remote-execution transports (SSH/ADB) for driving the executor on a remote device.
"""
import os
import shlex
import shutil
import socket
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
        """Stream `stdin` to `command` and return its raw stdout over adb's exec: service — no PTY and
        no device files, so the batch request/response crosses in one streamed round-trip."""
        cmd = f"su -c '{command}'" if privileged else command
        s = socket.create_connection((self.host, self.port), timeout=600)
        s.settimeout(600)
        try:
            def service(request: str) -> None:
                s.sendall(f"{len(request):04x}{request}".encode())
                status = s.recv(4)
                if status != b"OKAY":
                    raise IOError(f"adb rejected {request!r}: {status!r} {s.recv(4096)!r}")
            service(f"host:transport:{self.serial}")
            service(f"exec:{cmd}")
            s.sendall(stdin)
            chunks = []
            while True:
                chunk = s.recv(1 << 16)
                if not chunk:
                    break
                chunks.append(chunk)
            return b"".join(chunks)
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
