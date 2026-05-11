from enum import IntEnum
from logging import getLogger, Logger
from time import sleep
import numpy as np
from tqdm import tqdm

from api.src._helper import (
    FlashInfos
)

from api.mcu_api import (
    DeviceAPI,
    Commands as CmdsBasic
)


class Commands(IntEnum):
    FLASH_INIT = len(CmdsBasic) + 0x00
    FLASH_GET_INFOS = len(CmdsBasic) + 0x01
    FLASH_START_ERASE_ALL = len(CmdsBasic) + 0x02
    FLASH_START_ERASE_SEC = len(CmdsBasic) + 0x03
    FLASH_CHECK_ERASE = len(CmdsBasic) + 0x04
    FLASH_SET_ADDR_UPPER = len(CmdsBasic) + 0x05
    FLASH_SET_ADDR_LOWER = len(CmdsBasic) + 0x06
    FLASH_GET_ADDR = len(CmdsBasic) + 0x07
    FLASH_READ_DATA = len(CmdsBasic) + 0x08
    FLASH_WRITE_BUFFER = len(CmdsBasic) + 0x09
    FLASH_WRITE_DATA = len(CmdsBasic) + 0x0A
    FPGA_INIT = len(CmdsBasic) + 0x0B
    FPGA_POWER_STATE = len(CmdsBasic) + 0x0C
    FPGA_PROGRAM_STATE = len(CmdsBasic) + 0x0D
    FPGA_PROGRAM_CYCLE = len(CmdsBasic) + 0x0E
    FPGA_LOGIC_RESET = len(CmdsBasic) + 0x0F
    FPGA_TOGGLE_LED = len(CmdsBasic) + 0x10


class AccessFPGA(DeviceAPI):
    __logger: Logger

    def __init__(self, com_name: str = "AUTOCOM", timeout: float = 1.) -> None:
        """Interface class for handling with a custom DAQ device (with FPGA/Flash access)
        :param com_name:    String with the serial port name of the used device
        :param timeout:     Floating value with timeout for the communication [Default, not during DAQ]
        """
        super().__init__(com_name, timeout)
        self.__logger = getLogger(__name__)
        self.extend_pins_list(["FPGA_PWR_EN", "FPGA_PROGB", "FPGA_RSTN"])
        self.extend_states_list(["ERASE_FLASH"])

        self.__init_flash()
        self.__init_fpga()
        self.set_power_state(False)
        self.fpga_set_program_state(True)

    def __init_flash(self):
        ret = self._write_with_feedback(Commands.FLASH_INIT)
        if ret[1] != 1:
            raise ValueError("Initialization of FPGA failed!")

    def __init_fpga(self) -> None:
        ret = self._write_with_feedback(Commands.FPGA_INIT)
        if ret[1] != 1:
            raise ValueError("Initialization of FPGA failed!")

    def set_power_state(self, state: bool) -> None:
        """Setting the power state of the FPGA
        :param state:    Boolean value with the state of the FPGA (true = enabled, false = disabled)
        :return:        None
        """
        self._write_without_feedback(Commands.FPGA_POWER_STATE, int(state))
        sleep(0.5)

    @property
    def _package_flash_info(self) -> np.dtype:
        return np.dtype([
            ('manu', 'u1'),
            ('device', '<u2'),
            ('jedec', '<u2'),
            ('status', '<u2'),
            ('pages', '<u2'),
            ('blocks', '<u2'),
        ])

    def get_flash_infos(self) -> FlashInfos:
        """Getting the flash information from device
        :return:    Class FlashInfos with information about the flash
        """
        ret = self._write_with_feedback(Commands.FLASH_GET_INFOS, size=12)
        frame = np.frombuffer(ret, dtype=self._package_flash_info)[0]
        manu_id = int(frame['manu']) if frame['manu'] == (frame['device'] & 0x00FF) else 0
        return FlashInfos(
            manu_id=manu_id,
            dev_id=int(frame['device'] >> 8),
            mem_type=int(frame['jedec'] & 0x00FF),
            capacity=2 ** (int(frame['jedec'] >> 8)),
            status=int(frame['status']),
            pagesize=int(frame['pages']),
            blocksize=int(frame['blocks']),
        )

    def erase_flash_all(self) -> None:
        """Erasing all pages of the FPGA flash
        :return:    None
        """
        self._write_without_feedback(Commands.FLASH_START_ERASE_ALL)
        print("Start erasing the FPGA flash ...")
        while True:
            ret = self._write_with_feedback(Commands.FLASH_CHECK_ERASE)
            sleep(0.5)
            if ret[1] == 1:
                break
        if self.get_state().system == "ERASE_FLASH":
            raise SystemError("System state is not correct!")
        print("Erasing the FPGA flash is done ...")

    def erase_flash_sector(self, page: int, bytes_page: int) -> None:
        """Erasing a sector of the FPGA flash
        :param page:        Integer with the number of the page to erase
        :param bytes_page:  Integer with the number of bytes in each page
        :return:            None
        """
        self._set_flash_starting_address(page * bytes_page)
        self._write_without_feedback(Commands.FLASH_START_ERASE_SEC)
        while True:
            ret = self._write_with_feedback(Commands.FLASH_CHECK_ERASE)
            sleep(0.5)
            if ret[1] == 1:
                break
        if self.get_state().system == "ERASE_FLASH":
            raise SystemError("System state is not correct!")

    def _get_flash_starting_address(self) -> int:
        """Returning the starting address of the flash
        :return:    Integer with the starting address of the flash (bytes)
        """
        ret = self._write_with_feedback(Commands.FLASH_GET_ADDR, size=5)
        return int(self._bytes_to_int(ret, signed=False))

    def _set_flash_starting_address(self, addr: int) -> None:
        """Setting the starting address of the flash
        :param addr:    Integer with the starting address of the flash (bytes)
        :return:        None
        """
        if not 0 <= addr < 2 ** 32:
            raise ValueError("Address is out of range! [0, 2^32)")

        self._write_without_feedback(Commands.FLASH_SET_ADDR_UPPER, (addr >> 16) & 0xFFFF)
        self._write_without_feedback(Commands.FLASH_SET_ADDR_LOWER, addr & 0xFFFF)

    def _read_page_from_flash(self, page: int, bytes_page: int) -> list[int]:
        """Reading the page content from the flash
        :param page:        Integer with the number of the page to read
        :param bytes_page:  Integer with the number of bytes in each page
        :return:            List with the content of the page
        """
        self._set_flash_starting_address(page * bytes_page)

        data = self._write_with_feedback(Commands.FLASH_READ_DATA, size=bytes_page + 1)
        if len(data) != bytes_page:
            raise ValueError(f"Size of the readed data is not equal to byte_page ({bytes_page})")
        return list(data)

    def _write_page_into_flash(self, page: int, data: bytes, bytes_page: int, do_check: bool=False) -> None:
        """Writing the page content into the flash
        :param page:        Integer with the number of the page to write
        :param data:        Bytes with the content of the page
        :param bytes_page:  Integer with the number of bytes in each page
        :param do_check:    Boolean value with the check of the data after writing
        :return:            None
        """
        self._set_flash_starting_address(page * bytes_page)
        for a, b in zip(data[::2], data[1::2]):
            send_data = a * 256 + b
            self._write_without_feedback(Commands.FLASH_WRITE_BUFFER, send_data)
        ret = self._write_with_feedback(Commands.FLASH_WRITE_DATA)
        if ret[1] != 1:
            raise ValueError("Write process was not correct!")
        if do_check:
            data_check = bytes(self._read_page_from_flash(page, bytes_page))
            if data != data_check:
                raise ValueError("Data are unequal (Read data after writing is not equal!")

    @staticmethod
    def _split_stream_into_pages(bitstream: bytes, pagesize: int) -> list[bytes]:
        pages = []
        for idx in range(0, len(bitstream), pagesize):
            page = bitstream[idx:idx + pagesize]
            if len(page) < pagesize:
                page += b'\xFF' * (pagesize - len(page))
            pages.append(page)
        return pages

    def write_bitstream_into_flash(self, bitstream: bytes, start_page: int=0, do_check: bool=False) -> None:
        """Writing a bitstream into the FPGA flash
        :param bitstream:   Bytes with the bitstream to write
        :param start_page:  Integer with the number of the first page to write
        :param do_check:    Boolean value with the check of the data after writing
        :return:            None
        """
        sets = self.get_flash_infos()
        bitstream_chunks = self._split_stream_into_pages(bitstream, sets.pagesize)
        bitstream_chunks.append(bytes([255 for _ in range(sets.pagesize)]))
        bitstream_chunks.append(bytes([255 for _ in range(sets.pagesize)]))
        if len(bitstream_chunks) > sets.num_pages:
            raise ValueError("Bitstream is too long!")

        self.set_power_state(False)
        self.fpga_set_program_state(True)
        if "PROGB" in self.get_state().pins:
            raise ValueError("FPGA is not in init mode!")

        for page, data in enumerate(tqdm(bitstream_chunks, desc="Flashing the bitstream into FPGA flash: ")):
            sel_page = page + start_page
            if sel_page % sets.blocksize == 0:
                self.erase_flash_sector(sel_page, sets.pagesize)
            self._write_page_into_flash(sel_page, data, sets.pagesize, do_check)
        self.fpga_set_program_state(False)

    def check_bitstream_from_flash(self, bitstream: bytes, start_page: int=0) -> bool:
        """Checking if bitstream is equal to the content of the FPGA flash
        :param bitstream:   Bytes with the bitstream to check
        :param start_page:  Integer with the number of the first page to check
        :return:            Boolean value with the result of the check
        """
        sets = self.get_flash_infos()
        bitstream_chunks = self._split_stream_into_pages(bitstream, sets.pagesize)
        if len(bitstream_chunks) > sets.num_pages:
            raise ValueError("Bitstream is too long!")

        for page, data in enumerate(tqdm(bitstream_chunks, desc="Checking the bitstream in FPGA flash: ")):
            data_rd = bytes(self._read_page_from_flash(page + start_page, sets.pagesize))
            if data != data_rd:
                return False
        return True

    def fpga_set_program_state(self, state: bool) -> None:
        """Setting the program state of the FPGA
        :param state:    Boolean value with the state of the FPGA (true = active programming, false = init mode)
        :return:        None
        """
        self._write_without_feedback(Commands.FPGA_PROGRAM_STATE, int(state))

    def fpga_do_program_reset(self) -> None:
        """Resetting the FPGA program state
        :return:    None
        """
        self._write_without_feedback(Commands.FPGA_PROGRAM_CYCLE, 0)
        sleep(2)

    def fpga_do_logic_reset(self, num_iterations: int=1) -> None:
        """Do logic reset of the FPGA content
        :param num_iterations:      Integer with the number of iterations of the reset
        :return:                    None
        """
        if num_iterations < 1 or num_iterations > 255:
            raise ValueError("Number of iterations must be greater than [0, 256)!")
        self._write_without_feedback(Commands.FPGA_LOGIC_RESET, num_iterations)
        sleep(0.1)

    def fpga_toggle_led(self) -> None:
        """Toggling the LED of the FPGA
        :return:    None
        """
        ret = self._write_with_feedback(Commands.FPGA_TOGGLE_LED, data=0, size=4)
        print(ret)
