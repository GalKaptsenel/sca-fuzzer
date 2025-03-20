from ppadb.client import Client as AdbClient


class Connection:
    def __init__(self):
        pass

    def shell(self, command: str) -> str:
        raise NotImplementedError

    def push(self, src, dst):
        raise NotImplementedError

    def pull(self, src, dst):
        raise NotImplementedError


class USBConnection(Connection):
    def __init__(self, host, port=5037, serial: str = None):
        super(USBConnection, self).__init__()
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

    def shell(self, command: str) -> str:
        return self.device.shell(command)

    def push(self, src, dst):
        return self.device.push(src, dst)

    def pull(self, src, dst):
        return self.device.pull(src, dst)
