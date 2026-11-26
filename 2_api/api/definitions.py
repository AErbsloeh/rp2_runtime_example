from enum import IntEnum


class Commands(IntEnum):
    ECHO = 0x00
    RESET = 0x01
    GET_CHARAC_STATE = 0x02
    GET_NUMBER_DAQ = 0x03
    GET_CHARAC_DAQ = 0x04
    ENABLE_LED = 0x05
    DISABLE_LED = 0x06
    TOGGLE_LED = 0x07
    START_DAQ = 0x08
    STOP_DAQ = 0x09
    SET_PERIOD_DAQ = 0x0A
    PROCESS_DATA = 0x0B
    APPLY_DATA = 0x0C
    GET_CHARAC_PROCESSING = 0x0D


class DaqDataType(IntEnum):
    SAMPLE = 0x00
    CONT_WINDOW = 0x01
    EVENT_WINDOW = 0x02
    FEATURE = 0x03
    EVENT = 0x04
    CLASS = 0x05


class CommandsFPGA(IntEnum):
    FLASH_INIT = len(Commands) + 0x00
    FLASH_GET_INFOS = len(Commands) + 0x01
    FLASH_START_ERASE_ALL = len(Commands) + 0x02
    FLASH_START_ERASE_SEC = len(Commands) + 0x03
    FLASH_CHECK_ERASE = len(Commands) + 0x04
    FLASH_SET_ADDR_UPPER = len(Commands) + 0x05
    FLASH_SET_ADDR_LOWER = len(Commands) + 0x06
    FLASH_GET_ADDR = len(Commands) + 0x07
    FLASH_READ_DATA = len(Commands) + 0x08
    FLASH_WRITE_BUFFER = len(Commands) + 0x09
    FLASH_WRITE_DATA = len(Commands) + 0x0A
    FPGA_PROGRAM_STATE = len(Commands) + 0x0B
    FPGA_PROGRAM_CYCLE = len(Commands) + 0x0C
    FPGA_POWER_STATE = len(Commands) + 0x0D
    FPGA_LOGIC_RESET = len(Commands) + 0x0E
    FPGA_TOGGLE_LED = len(Commands) + 0x0F
    FPGA_RXTX_DATA = len(Commands) + 0x10
