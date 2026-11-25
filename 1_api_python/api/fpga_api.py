from logging import Logger, getLogger
from time import sleep

import numpy as np
from tqdm import tqdm

from .definitions import CommandsFPGA as Commands
from .mcu_api import DeviceAPI
from .src._helper import FlashInfos


class FlashFPGA(DeviceAPI):
    __logger: Logger
    _num_bytes_head: int
    _num_bytes_data: int
    _num_spi_middleware: tuple[int, int, int] = (2, 6, 16)  # (CMD, ADR, DATA) in bits

    def __init__(self, com_name: str = "AUTOCOM", timeout: float = 1.0) -> None:
        """Interface class for flashing the FPGA on a custom DAQ device
        :param com_name:    String with the serial port name of the used device
        :param timeout:     Floating value with timeout for the communication [Default, not during DAQ]
        """
        if not DeviceAPI.has_fpga:
            raise RuntimeError(
                "No FPGA is declared: set has_fpga = True in DeviceAPI if an FPGA is connected"
            )
        super().__init__(com_name, timeout)
        self.__logger = getLogger(__name__)
        self.extend_pins_list([])
        self.extend_states_list(["FLASHING", "ERASE_FLASH"])

        self.__init_fpga()
        self.set_fpga_power_state(False)
        self.fpga_set_program_state(True)

    def __init_fpga(self):
        ret = self._write_with_feedback(Commands.FLASH_INIT)
        if ret[1] != 1:
            raise ValueError("Initialization of FPGA failed!")

    @property
    def _package_flash_info(self) -> np.dtype:
        return np.dtype(
            [
                ("manu", "u1"),
                ("device", "<u2"),
                ("jedec", "<u2"),
                ("status", "<u2"),
                ("pages", "<u2"),
                ("blocks", "<u2"),
            ]
        )

    def get_flash_infos(self) -> FlashInfos:
        """Getting the flash information from device
        :return:    Class FlashInfos with information about the flash
        """
        ret = self._write_with_feedback(Commands.FLASH_GET_INFOS, size=12)
        frame = np.frombuffer(ret, dtype=self._package_flash_info)[0]
        manu_id = int(frame["manu"]) if frame["manu"] == (frame["device"] & 0x00FF) else 0
        return FlashInfos(
            manu_id=manu_id,
            dev_id=int(frame["device"] >> 8),
            mem_type=int(frame["jedec"] & 0x00FF),
            capacity=2 ** (int(frame["jedec"] >> 8)),
            status=int(frame["status"]),
            pagesize=int(frame["pages"]),
            blocksize=int(frame["blocks"]),
        )

    def erase_flash_all(self) -> None:
        """Erasing all pages of the FPGA flash
        :return:    None
        """
        self._write_without_feedback(Commands.FLASH_START_ERASE_ALL)
        print("Start erasing the FPGA flash ...")
        while True:
            ret = self._write_with_feedback(Commands.FLASH_CHECK_ERASE, size=2)
            sleep(0.25)
            if ret[0] == 1:
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
            ret = self._write_with_feedback(Commands.FLASH_CHECK_ERASE, size=2)
            sleep(0.1)
            if ret[0] == 1:
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
        if not 0 <= addr < 2**32:
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

    def _write_page_into_flash(
        self, page: int, data: bytes, bytes_page: int, do_check: bool = False
    ) -> None:
        """Writing the page content into the flash
        :param page:        Integer with the number of the page to write
        :param data:        Bytes with the content of the page
        :param bytes_page:  Integer with the number of bytes in each page
        :param do_check:    Boolean value with the check of the data after writing
        :return:            None
        """
        addr = page * bytes_page
        upper = (addr >> 16) & 0xFFFF
        lower = addr & 0xFFFF

        n = self._num_bytes_data
        rest = len(data) % n
        full = data[: len(data) - rest]
        tail = data[len(data) - rest :]

        out = bytearray()
        out.extend(upper.to_bytes(2, "little") + bytes(n - 2))
        out.append(Commands.FLASH_SET_ADDR_UPPER)
        out.extend(lower.to_bytes(2, "little") + bytes(n - 2))
        out.append(Commands.FLASH_SET_ADDR_LOWER)
        out.extend(
            x
            for i in range(0, len(full), n)
            for x in (*reversed(full[i : i + n]), Commands.FLASH_WRITE_BUFFER)
        )
        if tail:
            out.extend(b"\x00" * (n - len(tail)))
            out.extend(reversed(tail))
            out.append(Commands.FLASH_WRITE_BUFFER)
        out.extend((*([0] * self._num_bytes_data), Commands.FLASH_WRITE_DATA))

        ret = self._write_bytes(bytes(out), size=2)
        if ret[-1] != 1 and ret[0] != Commands.FLASH_WRITE_DATA:
            raise ValueError("Write process was not correct!")
        if do_check:
            data_check = bytes(self._read_page_from_flash(page, bytes_page))
            if data != data_check:
                raise ValueError(
                    f"Data are unequal!\nwrite: {data.hex(' ')}\n read: {data_check.hex(' ')}"
                )

    @staticmethod
    def _split_stream_into_pages(bitstream: bytes, pagesize: int) -> list[bytes]:
        pages = [bitstream[idx : idx + pagesize] for idx in range(0, len(bitstream), pagesize)]
        if not len(pages[-1]) == pagesize:
            last_page = pages[-1]
            last_page += bytes([0xFF for _ in range(pagesize - len(last_page))])
            pages[-1] = last_page
        return pages

    @staticmethod
    def _add_filling_bytes_into_pages(
        bitstream_pages: list[bytes], pagesize: int, num_ends: int
    ) -> list[bytes]:
        pages = bitstream_pages.copy()
        for _ in range(num_ends):
            pages.append(bytes([255 for _ in range(pagesize)]))
        return pages

    def write_bitstream_into_flash(
        self, bitstream: bytes, start_page: int = 0, do_check: bool = False
    ) -> None:
        """Writing a bitstream into the FPGA flash
        :param bitstream:   Bytes with the bitstream to write
        :param start_page:  Integer with the number of the first page to write
        :param do_check:    Boolean value with the check of the data after writing
        :return:            None
        """
        sets = self.get_flash_infos()
        bitstream_chunks = self._split_stream_into_pages(bitstream, sets.pagesize)
        bitstream_chunks = self._add_filling_bytes_into_pages(bitstream_chunks, sets.pagesize, 4)

        if len(bitstream_chunks) > sets.num_pages:
            raise ValueError("Bitstream is too long!")

        self.set_fpga_power_state(False)
        self.fpga_set_program_state(True)
        if "PROGB" in self.get_state().pins:
            raise ValueError("FPGA is not in init mode!")

        for page, data in enumerate(
            tqdm(bitstream_chunks, desc="Flashing the bitstream into FPGA flash: ")
        ):
            sel_page = page + start_page
            if sel_page % sets.blocksize == 0:
                self.erase_flash_sector(sel_page, sets.pagesize)
            self._write_page_into_flash(sel_page, data, sets.pagesize, do_check)
        self.fpga_set_program_state(False)

    def check_bitstream_from_flash(self, bitstream: bytes, start_page: int = 0) -> bool:
        """Checking if bitstream is equal to the content of the FPGA flash
        :param bitstream:   Bytes with the bitstream to check
        :param start_page:  Integer with the number of the first page to check
        :return:            Boolean value with the result of the check
        """
        sets = self.get_flash_infos()
        data_rd = bytes()
        for page, data in enumerate(
            tqdm(
                range(int(len(bitstream) / sets.pagesize)), desc="Checking the bitstream in FPGA flash: "
            )
        ):
            data_rd += bytes(self._read_page_from_flash(page + start_page, sets.pagesize))

        for i, (a, b) in enumerate(zip(bitstream, data_rd)):
            if a != b:
                print(f"Error @ byte {i}: expected=0x{a:02X}, get=0x{b:02X}")
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

    def set_fpga_power_state(self, state: bool) -> None:
        """Setting the power state of the FPGA
        :param state:    Boolean value with the state of the FPGA (true = enabled, false = disabled)
        :return:        None
        """
        self._write_without_feedback(Commands.FPGA_POWER_STATE, int(state))
        sleep(0.5)

    def fpga_do_logic_reset(self, num_iterations: int = 1) -> None:
        """Do logic reset of the FPGA content
        :param num_iterations:      Integer with the number of iterations of the reset
        :return:                    None
        """
        if num_iterations < 1 or num_iterations > 255:
            raise ValueError("Number of iterations must be greater than [0, 256)!")
        self._write_without_feedback(Commands.FPGA_LOGIC_RESET, num_iterations)
        sleep(0.1)

    def _convert_raw_into_middleware(self, cmd: int, adr: int, data: int) -> int:
        """Packing cmd/adr/data into the SPI frame of the middleware
        Frame: [CMD | ADR | DATA] (MSB first), widths from self._num_spi_middleware
        :param cmd:     Integer with the command
        :param adr:     Integer with the address
        :param data:    Integer with the data
        :return:        Integer with the complete frame
        """
        bits_cmd, bits_adr, bits_data = self._num_spi_middleware
        for name, val, width in (
            ("cmd", cmd, bits_cmd),
            ("adr", adr, bits_adr),
            ("data", data, bits_data),
        ):
            if not 0 <= val < 2**width:
                raise ValueError(f"{name} must be in [0, {2**width}) ({width} bit)")
        return (cmd << (bits_adr + bits_data)) | (adr << bits_data) | data

    def fpga_toggle_led(self) -> None:
        """Toggling the LED of the FPGA
        :return:    None
        """
        data = self._convert_raw_into_middleware(cmd=0, adr=8, data=0)
        self._write_without_feedback(Commands.FPGA_TOGGLE_LED, data=data)

    def fpga_control_led(self, state: bool) -> None:
        """Controlling the LED of the FPGA
        :param:     State of the LED
        :return:    None
        """
        data = self._convert_raw_into_middleware(cmd=0, adr=4, data=int(state))
        self._write_without_feedback(Commands.FPGA_TOGGLE_LED, data=data)

    def fpga_send_data(self, data: bytes) -> bytes:
        """Instruction to send data from host computer to the FPGA and return data.
        :param data:    Bytes with data content to the FPGA
        :return:        Bytes with data from the FPGA
        """
        num_bytes_frame = (sum(self._num_spi_middleware) + 7) // 8
        return self._write_with_feedback(
            head=Commands.FPGA_RXTX_DATA,
            data=int.from_bytes(bytes=data, byteorder="big", signed=False),
            size=1 + num_bytes_frame,
        )
