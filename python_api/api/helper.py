from dataclasses import dataclass


def get_path_to_project(new_folder: str='', max_levels: int=5) -> str:
    """Function for getting the path to find the project folder structure in application.
    :param new_folder:  New folder path
    :param max_levels:  Max number of levels to get-out for finding pyproject.toml
    :return:            String with an absolute path to start the project structure
    """
    from pathlib import Path
    cwd = Path(".").absolute()
    current = cwd

    def is_project_root(p):
        return (p / "pyproject.toml").exists()

    for _ in range(max_levels):
        if is_project_root(current):
            return str(current / new_folder)
        current = current.parent

    if is_project_root(current):
        return str(current / new_folder)
    return str(cwd)


@dataclass(frozen=True)
class DataAcquisitionConfig:
    """Dataclass with configuration of the Data Acquisition Unit
    Attributes:
        send_batch:       Boolean for sending data in batch mode
        sampling_rate:    Float with sampling rate [Hz]
        num_channels:     Integer with number of channels
        num_samples:      Integer with number of samples per channel
        head_cmd:         Integer with header command of data package
        tail_cmd:         Integer with tail command of data package
        num_bytes_total:  Integer with total number of bytes for each transmission
        bytes_sample:     Integer with number of bytes for each sample
        is_signed:        Boolean if data sample is signed or unsigned
    """
    send_batch: bool
    sampling_rate: float
    num_channels: int
    num_samples: int
    head_cmd: int
    tail_cmd: int
    num_bytes_total: int
    bytes_sample: int
    is_signed: bool

    @property
    def data_shape(self) -> tuple:
        """Returning tuple with shape of the data array during data acquisition (channels, samples)"""
        if self.send_batch:
            return self.num_channels, self.num_samples
        else:
            return self.num_channels,

    @property
    def dtype_sample(self) -> str:
        """Returning string with numpy data type definition for each data sample during data acquisition"""
        datatype = 'i' if self.is_signed else 'u'
        return f'<{datatype}{self.bytes_sample}'


@dataclass(frozen=True)
class SystemState:
    """Dataclass for handling the system state of the device
    Attributes:
        pins:       String with enabled GPIO pins (selection from MCU)
        system:     String with actual system state
        runtime:    Float with actual execution runtime after last reset [sec.]
        clock:      Integer with System Clock [kHz]
        firmware:   String with a firmware version on board
        temp:       Float with temperature [°C] of the board
    """
    pins: str
    system: str
    runtime: float
    clock: int
    firmware: str
    temp: float


def _convert_pin_state(state: int, pin_list: list[str]) -> str:
    """Function for converting the pin state
    :param state:       Integer with pin state from MCU
    :param pin_list:    List with stings of pin names in right order, defined in the firmware
    :return:            String with pin state
    """
    if state == 0:
        return 'NONE'
    else:
        ret_text = ''
        for idx, led in enumerate(pin_list):
            if state & (1 << idx):
                ret_text += f'{led}' if len(ret_text) == 0 else f'+{led}'
        if ret_text == '':
            raise ValueError("Translated pin state is undefined")
        return ret_text


def _convert_system_state(state: int, state_list: list[str]) -> str:
    """Function for converting the pin state
    :param state:           Integer with pin state from MCU
    :param state_list:      List with stings of pin names in right order, defined in the firmware
    :return:                String with pin state
    """
    if not 0 <= state < len(state_list):
        raise ValueError(f'Invalid system state: {state}')
    return state_list[state]


def _convert_rp2_adc_value(raw: int) -> float:
    """Function for converting the RP2 ADC value from integer to float"""
    if raw >= 4095:
        val0 = 4095
    elif raw < 0:
        val0 = 0
    else:
        val0 = raw
    return val0 * 3.3 / 4095


def _convert_rp2_temp_value(raw: int) -> float:
    """Function for converting the RP2 temperatur value from integer to float"""
    volt = _convert_rp2_adc_value(raw)
    return 27 - (volt - 0.706) / 0.001721
