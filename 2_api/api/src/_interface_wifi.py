from logging import Logger, getLogger
from socket import (
    IPPROTO_TCP,
    TCP_NODELAY,
    create_connection,
    socket,
)
from socket import (
    timeout as SocketTimeout,
)
from typing import Optional


class InterfaceWifi:
    __logger: Logger
    __device: Optional[socket]
    __BYTES_HEAD: int
    __BYTES_DATA: int
    __timeout: float

    def __init__(
        self,
        host: str,
        port: int = 4242,
        num_bytes_head: int = 1,
        num_bytes_data: int = 2,
        timeout: float = 1.0,
    ) -> None:
        """Class for interacting with TCP devices using WiFi
        :param host:           String with IP address or hostname of the device
        :param port:           Integer with TCP port of the device
        :param num_bytes_head: Number of bytes head, implemented on Pico
        :param num_bytes_data: Number of bytes data, implemented on Pico
        :param timeout:        Floating value with timeout for the communication
        """
        self.__logger = getLogger(__name__)
        self.__host = host
        self.__port = port
        self.__BYTES_HEAD = num_bytes_head
        self.__BYTES_DATA = num_bytes_data
        self.__timeout = timeout
        self.__device = None

    @property
    def is_open(self) -> bool:
        """Return True if the device is open, False otherwise"""
        return self.__device is not None

    @property
    def total_num_bytes(self) -> int:
        """Returning the total number of bytes for each transmission"""
        return self.__BYTES_DATA + self.__BYTES_HEAD

    @property
    def num_bytes(self) -> int:
        """Returning the number of data bytes in each transmission"""
        return self.__BYTES_DATA

    @property
    def timeout(self) -> float:
        """Returning the communication timeout"""
        return self.__timeout

    @timeout.setter
    def timeout(self, value: float) -> None:
        """Setting the communication timeout"""
        self.__timeout = value
        if self.__device is not None:
            self.__device.settimeout(value)

    def convert(self, head: int, data: int) -> bytes:
        """Function for converting data to bytes
        :param head:    Header information, RPC command
        :param data:    Data to be converted
        :return:        Bytes converted from head + data
        """
        transmit = data.to_bytes(self.__BYTES_DATA, byteorder="little")
        transmit += head.to_bytes(self.__BYTES_HEAD, byteorder="little")
        return transmit

    def fileno(self) -> int:
        """File descriptor of the opened socket (for the C reader)"""
        return self._get_device().fileno()

    def _get_device(self) -> socket:
        if self.__device is None:
            raise ConnectionError("WiFi socket is not open")
        return self.__device

    def read(self, no_bytes: int) -> bytes:
        """Read content from device"""
        data = bytes()
        device = self._get_device()

        while len(data) < no_bytes:
            try:
                chunk = device.recv(no_bytes - len(data))
            except SocketTimeout:
                break

            if not chunk:
                break

            data += chunk

        return data

    def write(self, data: bytes) -> None:
        """Write content to device without feedback"""
        self._get_device().sendall(data)

    def write_wfb(self, data: bytes, size: int = 0) -> bytes:
        """Write all information to device (specific bytes)"""
        self.write(data)
        return self.read(len(data) if size <= 0 else size)

    def write_wfb_lf(self, data: bytes) -> bytes:
        """Write all information to device (unlimited bytes until LF)"""
        self.write(data)

        ret = bytes()
        while True:
            chunk = self.read(1)
            if not chunk:
                break
            ret += chunk
            if chunk == b"\n":
                break
        return ret

    def serialize_string(self, data: str) -> list:
        n = self.__BYTES_DATA
        raw = data.encode("utf-8")
        raw += b" " * ((-len(raw)) % n)
        return [int.from_bytes(raw[i : i + n], "big") for i in range(0, len(raw), n)]

    @staticmethod
    def deserialize_string(data: bytes, length: int) -> str:
        return data[:length].decode("utf-8")

    def open(self) -> None:
        """Starting a connection to device"""
        if self.__device is not None:
            self.close()

        self.__device = create_connection(
            (self.__host, self.__port),
            timeout=self.__timeout,
        )
        self.__device.setsockopt(IPPROTO_TCP, TCP_NODELAY, 1)
        self.__device.settimeout(self.__timeout)

    def close(self) -> None:
        """Closing a connection to device"""
        if self.__device is not None:
            self.__device.close()
            self.__device = None

    def empty_buffer(self) -> int:
        """Function for emptying input buffer
        :return:    Integer with number of discarded input bytes
        """
        device = self._get_device()
        old_timeout = self.__timeout
        device.settimeout(0.0)
        num_bytes = 0

        while True:
            try:
                data = device.recv(4096)
            except (BlockingIOError, SocketTimeout):
                break

            if not data:
                break
            num_bytes += len(data)

        device.settimeout(old_timeout)
        return num_bytes
