"""
File: AArch64 remote-execution transports (SSH/ADB) for driving the executor on a remote device.
"""
from ppadb.client import Client as AdbClient
import paramiko

from .aarch64_kernel import profile_op


class Connection:
    def __init__(self):
        pass

    def shell(self, command: str, privileged = False) -> str:
        raise NotImplementedError

    def push(self, src, dst):
        raise NotImplementedError

    def pull(self, src, dst):
        raise NotImplementedError

    def is_file_present(self, filename: str) -> bool:
        raise NotImplementedError


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
        cmd = f'sudo -S bash -c "{cmd}"' if privileged else cmd
        stdin, stdout, stderr = self.client.exec_command(cmd)

        if privileged and stdin is not None:
            password = self.password if self.password else ""
            stdin.write(password + "\n")
            stdin.flush()

        out = stdout.read().decode().strip()
        err = stderr.read().decode().strip()
        return out if out else err

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
            for device in self.client.devices():
                if device.serial == serial:
                    self.device = device
        else:
            self.device = self.client.devices()[0]


    def shell(self, cmd: str, privileged = False) -> str:
        cmd = f'su -c "{cmd}"' if privileged else cmd
        return self.device.shell(cmd)

    def push(self, src, dst):
        return self.device.push(src, dst)

    def pull(self, src, dst):
        return self.device.pull(src, dst)

    def is_file_present(self, filename: str) -> bool:
        return 'No such file or directory' not in self.shell(f'ls {filename}')
