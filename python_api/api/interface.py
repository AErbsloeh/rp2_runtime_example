from logging import Logger, getLogger
from serial import (
    Serial,
    PARITY_NONE,
    STOPBITS_ONE,
    EIGHTBITS
)
from serial.tools import list_ports


def get_comport_name(usb_vid: int) -> str:
    """Returning the COM Port name of the addressable devices
    :param usb_vid: USB VID
    :return:        String with COM port name with matched VIP und PID properties
    """
    available_ports = list_ports.comports()
    list_right_com = [port.device for port in available_ports if port.vid == usb_vid and (port.pid == 0x000A or port.pid == 0x0009)]
    if len(list_right_com) == 0:
        raise ConnectionError(f"No COM Port with right USB found - Please adapt the VID and PID values {[[port.name, port.vid, port.pid] for port in list_ports.comports()]}")
    return list_right_com[0]


class InterfaceSerial:
    __logger: Logger
    __device: Serial
    __BYTES_HEAD: int
    __BYTES_DATA: int

    def __init__(self, com_name: str, baud: int=115200, num_bytes_head: int=1, num_bytes_data: int=2, timeout: float=1.) -> None:
        """Class for interacting with the USB serial devices
        :param com_name:        String with name of the COM port to the device
        :param baud:            Integer with BAUDRATE for the communication between host and device
        :param num_bytes_head: Number of bytes head, implemented on Pico
        :param num_bytes_data: Number of bytes data, implemented on Pico
        """
        self.__logger = getLogger(__name__)
        self.__BYTES_HEAD = num_bytes_head
        self.__BYTES_DATA = num_bytes_data
        self.__device = Serial(
                port=com_name,
                baudrate=baud,
                parity=PARITY_NONE,
                stopbits=STOPBITS_ONE,
                bytesize=EIGHTBITS,
                inter_byte_timeout=timeout,
                xonxoff=False,
                rtscts=False,
                dsrdtr=False,
                timeout=2*timeout
            )

    @property
    def total_num_bytes(self) -> int:
        """Returning the total number of bytes for each transmission"""
        return self.__BYTES_DATA + self.__BYTES_HEAD

    @property
    def num_bytes(self) -> int:
        """Returning the number of data bytes in each transmission"""
        return self.__BYTES_DATA

    def convert(self, head: int, data: int) -> bytes:
        """Function for converting data to bytes
        :param head:    Header information, RPC command
        :param data:    Data to be converted
        :return:        Bytes converted from head + data
        """
        transmit = data.to_bytes(self.__BYTES_DATA, 'little')
        transmit += head.to_bytes(self.__BYTES_HEAD, 'little')
        return transmit

    def is_open(self) -> bool:
        """Return True if the device is open, False otherwise"""
        return self.__device.is_open

    def read(self, no_bytes: int) -> bytes:
        """Read content from device"""
        return self.__device.read(no_bytes)

    def write(self, data: bytes) -> None:
        """Write content to device without feedback"""
        self.__device.write(data)

    def write_wfb(self, data: bytes, size:int=0) -> bytes:
        """Write all information to device (specific bytes)"""
        num = self.__device.write(data)
        return self.__device.read(num if size <= 0 else size)

    def write_wfb_lf(self, data: bytes) -> bytes:
        """Write all information to device (unlimited bytes until LF)"""
        self.__device.write(data)
        return self.__device.read_until()

    @staticmethod
    def serialize_string(data: str, do_padding: bool) -> list:
        """Serialize a string to bytes"""
        if do_padding:
            data += " "
        chunks = [int.from_bytes(data[i:i + 2].encode('utf-8'), 'big') for i in range(0, len(data), 2)]
        return chunks

    @staticmethod
    def deserialize_string(data: bytes, do_padding: bool) -> str:
        val = data if not do_padding else data[:-1]
        return val.decode('utf8')

    def open(self) -> None:
        """Starting a connection to device"""
        if self.__device.is_open:
            self.__device.close()
        self.__device.open()

    def close(self) -> None:
        """Closing a connection to device"""
        self.__device.close()

    def empty_buffer(self) -> None:
        """Function for emptying input and output buffers"""
        self.__device.reset_input_buffer()
        self.__device.reset_output_buffer()
