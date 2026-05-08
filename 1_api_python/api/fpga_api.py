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
    INIT_FPGA = len(CmdsBasic) + 0x00
    SET_POWER_STATE = len(CmdsBasic) + 0x01
    GET_FLASH_INFOS = len(CmdsBasic) + 0x02
    START_ERASE_FLASH_ALL = len(CmdsBasic) + 0x03
    START_ERASE_FLASH_SEC = len(CmdsBasic) + 0x04
    CHECK_ERASE_FLASH = len(CmdsBasic) + 0x05
    SET_FLASH_ADDR_UPPER = len(CmdsBasic) + 0x06
    SET_FLASH_ADDR_LOWER = len(CmdsBasic) + 0x07
    GET_FLASH_ADDR = len(CmdsBasic) + 0x08
    READ_FLASH_DATA = len(CmdsBasic) + 0x09
    WRITE_FLASH_BUFFER = len(CmdsBasic) + 0x0A
    WRITE_FLASH_DATA = len(CmdsBasic) + 0x0B
    FPGA_DO_RESET = len(CmdsBasic) + 0x0C
    FPGA_TOGGLE_LED = len(CmdsBasic) + 0x0D


class AccessFPGA(DeviceAPI):
    __logger: Logger

    def __init__(self, com_name: str = "AUTOCOM", timeout: float = 1.) -> None:
        """Interface class for handling with a custom DAQ device (with FPGA/Flash access)
        :param com_name:    String with the serial port name of the used device
        :param timeout:     Floating value with timeout for the communication [Default, not during DAQ]
        """
        super().__init__(com_name, timeout)
        self.__logger = getLogger(__name__)
        self.extend_pins_list(["FPGA_PWR_EN"])
        self.extend_states_list(["ERASE_FLASH"])
        self._init_fpga_pins()

    def _init_fpga_pins(self) -> None:
        ret = self._write_with_feedback(Commands.INIT_FPGA)
        if ret[1] != 1:
            raise ValueError("Initialization of FPGA failed!")

    def set_power_state(self, state: bool) -> None:
        """"""
        self._write_without_feedback(Commands.SET_POWER_STATE, int(state))
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
        """"""
        ret = self._write_with_feedback(Commands.GET_FLASH_INFOS, size=12)
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

    def do_erase_flash_all(self) -> None:
        """"""
        self._write_without_feedback(Commands.START_ERASE_FLASH_ALL)
        print("Start erasing the FPGA flash ...")
        while True:
            ret = self._write_with_feedback(Commands.CHECK_ERASE_FLASH)
            sleep(0.5)
            if ret[1] == 1:
                break
        if self.get_state().system == "ERASE_FLASH":
            raise SystemError("System state is not correct!")
        print("Erasing the FPGA flash is done ...")

    def do_erase_flash_sector(self, page: int, bytes_page: int) -> None:
        """"""
        self.set_flash_starting_address(page * bytes_page)
        self._write_without_feedback(Commands.START_ERASE_FLASH_SEC)
        while True:
            ret = self._write_with_feedback(Commands.CHECK_ERASE_FLASH)
            sleep(0.5)
            if ret[1] == 1:
                break
        if self.get_state().system == "ERASE_FLASH":
            raise SystemError("System state is not correct!")

    def get_flash_starting_address(self) -> int:
        """"""
        ret = self._write_with_feedback(Commands.GET_FLASH_ADDR, size=5)
        return int(self._bytes_to_int(ret, signed=False))

    def set_flash_starting_address(self, addr: int) -> None:
        """"""
        if not 0 <= addr < 2 ** 32:
            raise ValueError("Address is out of range! [0, 2^32)")

        self._write_without_feedback(Commands.SET_FLASH_ADDR_UPPER, (addr >> 16) & 0xFFFF)
        self._write_without_feedback(Commands.SET_FLASH_ADDR_LOWER, addr & 0xFFFF)

    def read_page_from_flash(self, page: int, bytes_page: int) -> list:
        """"""
        self.set_flash_starting_address(page * bytes_page)

        data = self._write_with_feedback(Commands.READ_FLASH_DATA, size=bytes_page+1)
        if len(data) != bytes_page:
            raise ValueError("Read data is not correct!")
        return list(data)

    def write_page_into_flash(self, page: int, data: bytes, bytes_page: int, do_check: bool=True) -> None:
        """"""
        self.set_flash_starting_address(page * bytes_page)
        for a, b in zip(data[::2], data[1::2]):
            send_data = a * 256 + b
            self._write_without_feedback(Commands.WRITE_FLASH_BUFFER, send_data)
        ret = self._write_with_feedback(Commands.WRITE_FLASH_DATA)
        if ret[1] != 1:
            raise ValueError("Write process was not correct!")
        if do_check:
            data_check = bytes(self.read_page_from_flash(page, bytes_page))
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
        """"""
        sets = self.get_flash_infos()
        bitstream_chunks = self._split_stream_into_pages(bitstream, sets.pagesize)
        if len(bitstream_chunks) > sets.num_pages:
            raise ValueError("Bitstream is too long!")

        for page, data in enumerate(tqdm(bitstream_chunks)):
            sel_page = page + start_page
            if sel_page % sets.blocksize == 0:
                self.do_erase_flash_sector(sel_page, sets.pagesize)
            self.write_page_into_flash(sel_page, data, sets.pagesize, do_check)

    def check_bitstream_from_flash(self, bitstream: bytes, start_page: int=0) -> bool:
        sets = self.get_flash_infos()
        bitstream_chunks = self._split_stream_into_pages(bitstream, sets.pagesize)
        if len(bitstream_chunks) > sets.num_pages:
            raise ValueError("Bitstream is too long!")

        for page, data in enumerate(tqdm(bitstream_chunks)):
            data_rd = bytes(self.read_page_from_flash(page + start_page, sets.pagesize))
            if data != data_rd:
                return False
        return True

    def fpga_do_reset(self, num_iterations: int=1) -> None:
        self._write_without_feedback(Commands.FPGA_DO_RESET, num_iterations)

    def fpga_toggle_led(self) -> None:
        self._write_without_feedback(Commands.FPGA_TOGGLE_LED)