from dataclasses import dataclass
from enum import IntEnum
from logging import getLogger, Logger
from time import sleep
import numpy as np

from api.interface import (
    get_comport_name,
    InterfaceSerial
)
from api.lsl import ThreadLSL
from api.mcu_conv import (
    _convert_pin_state,
    _convert_system_state,
    _convert_rp2_temp_value
)


def get_path_to_project(new_folder: str='', max_levels: int=5) -> str:
    """Function for getting the path to find the project folder structure in application.
    :param new_folder:  New folder path
    :param max_levels:  Max number of levels to get-out for finding pyproject.toml
    :return:            String with absolute path to start the project structure
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
class SystemState:
    """Dataclass for handling the system state of the device
    Attributes:
        pins:       String with enabled GPIO pins (selection from MCU)
        system:     String with actual system state
        runtime:    Float with actual execution runtime after last reset [sec.]
        clock:      Integer with System Clock [kHz]
        firmware:   String with firmware version on board
        temp:       Float with temperature [°C] of the board
    """
    pins: str
    system: str
    runtime: float
    clock: int
    firmware: str
    temp: float
    

class Commands(IntEnum):
    """Enum Class for defining the address of the Remote Procedure Calls (RPC)"""
    ECHO = 0
    RESET = 1
    GET_CLOCK = 2
    GET_STATE = 3
    GET_PINS = 4
    GET_RUNTIME = 5
    GET_VERSION = 6
    GET_TEMP = 7
    ENABLE_LED = 8
    DISABLE_LED = 9
    TOGGLE_LED = 10
    START_SAMPLE_DAQ = 11
    START_BATCH_DAQ = 12
    STOP_DAQ = 13
    UPDATE_DAQ = 14
    ASK_BATCH_DAQ = 15


class DeviceAPI:
    __device: InterfaceSerial
    __threads: ThreadLSL
    __logger: Logger
    __timeout_default: float = 10.
    __num_samples_batch: int = 8
    __num_bytes_sample: int = 0
    __num_bytes_batch: int = 0
    __sampling_rate: float = 4.
    __usb_vid: int = 0x2E8A
    # PID of RP2350 = 0x0009 and RP2040 = 0x000A

    def __init__(self, com_name: str="AUTOCOM", timeout: float=1.) -> None:
        """Init. of the device with name and baudrate of the device
        :param com_name:    String with the serial port name of the used device
        :param timeout:     Floating value with timeout for the communication [Default, not during DAQ]
        """
        self.__logger = getLogger(__name__)
        self.__threads = ThreadLSL()
        self.__timeout_default = timeout
        self.__define_bytes_data(self.__num_samples_batch)
        self.__device = InterfaceSerial(
            com_name=com_name if com_name != "AUTOCOM" else get_comport_name(usb_vid=self.__usb_vid),
            baud=230400,
            num_bytes_head=1,
            num_bytes_data=2,
            timeout=self.__timeout_default
        )
        if self.is_com_port_active:
            self.__device.close()
        self.__device.open()

    def __write_with_feedback(self, head: Commands, data: int, size: int=0) -> bytes:
        return self.__device.write_wfb(
            data=self.__device.convert(head, data),
            size=size
        )

    def __write_without_feedback(self, head: Commands, data: int) -> None:
        self.__device.write(
            data=self.__device.convert(head, data),
        )

    def __define_bytes_data(self, samples_per_batch: int) -> None:
        self.__num_samples_batch = samples_per_batch
        self.__num_bytes_sample: int = 11 + 4
        self.__num_bytes_batch: int = 11 + 4 * samples_per_batch

    @property
    def total_num_bytes(self) -> int:
        """Returning the total number of bytes for each transmission"""
        return self.__device.total_num_bytes

    @property
    def is_com_port_active(self) -> bool:
        """Boolean for checking if serial communication is open and used"""
        return self.__device.is_open()

    @property
    def is_daq_running(self) -> bool:
        """Returning if DAQ is still running"""
        return self._get_system_state() == "DAQ" and self.__threads.is_alive

    def open(self) -> None:
        """Opening the serial communication between API and device"""
        self.__device.open()

    def close(self) -> None:
        """Closing the serial communication between API and device"""
        self.__device.close()

    def do_reset(self) -> None:
        """Performing a Software Reset on the Platform"""
        if self.__threads.is_alive:
            self.__threads.stop()
            self.stop_daq()
        self.__write_without_feedback(Commands.RESET, 0)
        sleep(4)

    def echo(self, data: str) -> str:
        """Sending some characters to the device and returning the result
        :param data:    String with the data to be sent
        :return:        String with returned data from DAQ
        """
        do_padding = len(data) % self.__device.num_bytes == 1
        val = bytes()
        for chunk in self.__device.serialize_string(data, do_padding):
            ret = self.__write_with_feedback(Commands.ECHO, chunk)
            if ret[0] != Commands.ECHO:
                raise ValueError(f"Get: {ret}")
            val += ret[1:]
        return self.__device.deserialize_string(val, do_padding)

    def _get_system_clock_khz(self) -> int:
        """Returning the system clock of the device in kHz"""
        ret = self.__write_with_feedback(Commands.GET_CLOCK, 0)
        if ret[0] != Commands.GET_CLOCK:
            raise ValueError(f"Get: {ret}")
        return 10 * int.from_bytes(ret[1:], byteorder='little', signed=False)

    def _get_system_state(self) -> str:
        """Retuning the System State"""
        ret = self.__write_with_feedback(Commands.GET_STATE, 0)
        if ret[0] != Commands.GET_STATE:
            raise ValueError(f"Get: {ret}")
        return _convert_system_state(ret[-1])

    def _get_pin_state(self) -> str:
        """Retuning the Pin States"""
        ret = self.__write_with_feedback(Commands.GET_PINS, 0)
        if ret[0] != Commands.GET_PINS:
            raise ValueError(f"Get: {ret}")
        return _convert_pin_state(ret[-1])

    def _get_runtime_sec(self) -> float:
        """Returning the execution runtime of the device after last reset
        :return:    Float value with runtime in seconds
        """
        ret = self.__write_with_feedback(Commands.GET_RUNTIME, 0, size=9)
        if ret[0] != Commands.GET_RUNTIME:
            raise ValueError(f"Get: {ret}")
        return 1e-6 * int.from_bytes(ret[1:], byteorder='little', signed=False)

    def _get_firmware_version(self) -> str:
        """Returning the firmware version of the device
        :return:    String with firmware version
        """
        ret = self.__write_with_feedback(Commands.GET_VERSION, 0)
        if ret[0] != Commands.GET_VERSION:
            raise ValueError(f"Get: {ret}")
        return f"{ret[1]}.{ret[2]}"

    def _get_temp_mcu(self) -> float:
        """Returning the temperature of the device in Celsius
        :return:    Float value with temperature in Celsius
        """
        ret = self.__write_with_feedback(Commands.GET_TEMP, 0)
        if ret[0] != Commands.GET_TEMP:
            raise ValueError(f"Get: {ret}")
        return _convert_rp2_temp_value(int.from_bytes(ret[1:], signed=False, byteorder='little'))

    def get_state(self) -> SystemState:
        """Returning the state of the system
        :return:    Class SystemState with information about pin state, system state and actual runtime of the system
        """
        return SystemState(
            pins=self._get_pin_state(),
            system=self._get_system_state(),
            runtime=self._get_runtime_sec(),
            clock=self._get_system_clock_khz(),
            firmware=self._get_firmware_version(),
            temp=self._get_temp_mcu()
        )

    def enable_led(self) -> None:
        """Changing the state of the LED with enabling it
        :return:        None
        """
        self.__write_without_feedback(Commands.ENABLE_LED, 0)

    def disable_led(self) -> None:
        """Changing the state of the LED with disabling it
        :return:        None
        """
        self.__write_without_feedback(Commands.DISABLE_LED, 0)

    def toggle_led(self) -> None:
        """Changing the state of the LED with toggling it
        :return:        None
        """
        self.__write_without_feedback(Commands.TOGGLE_LED, 0)

    @property
    def _thread_frame_datatype(self) -> np.dtype:
        return np.dtype([
            ('head', 'u1'),         # 1 Byte unsigned
            ('index', 'u1'),        # 1 Byte unsigned
            ('timestamp', '<u8'),   # 8 Byte unsigned
            ('data', '<u2', (2,)),  # 2 Byte signed short (channels, samples)
            ('tail', 'u1')          # 1 Byte unsigned
        ])

    @property
    def _thread_batch_datatype(self) -> np.dtype:
        return np.dtype([
            ('head', 'u1'),         # 1 Byte unsigned
            ('index', 'u1'),        # 1 Byte unsigned
            ('timestamp', '<u8'),   # 8 Byte unsigned
            ('data', '<u2', (2,self.__num_samples_batch,)),# 2 Byte signed short (channels, samples)
            ('tail', 'u1')          # 1 Byte unsigned
        ])

    def _thread_read_frame(self) -> tuple[list, float]:
        """Entpacken der Informationen aus dem USB Protokoll (siehe C-Datei: src/daq_sample.c in der Firmware)"""
        try:
            buffer = self.__device.read(self.__num_bytes_sample)
            if not buffer:
                raise Exception
            frames = np.frombuffer(buffer, dtype=self._thread_frame_datatype)
            mask = (frames['head'] == 0xA0) & (frames['tail'] == 0xFF)
            frames = frames[mask]

            if frames.size > 0:
                timestamps = float(frames['timestamp'][0] * 1e-6)
                data = frames['data'][0].tolist()
                return data, timestamps
            else:
                raise Exception
        except Exception:
            return [], None

    def _thread_read_batch(self) -> tuple[list[list], list[float]]:
        """Entpacken der Informationen aus dem USB Protokoll (siehe C-Datei: src/daq_sample.c in der Firmware)"""
        try:
            buffer = self.__device.read(self.__num_bytes_batch)
            if not buffer:
                raise Exception
            frames = np.frombuffer(buffer, dtype=self._thread_batch_datatype)
            mask = (frames['head'] == 0xA0) & (frames['tail'] == 0xFF)
            frames = frames[mask]
            if frames.size > 0:
                timestamps = [float(1e-6 * frames['timestamp'][0]) for _ in range(self.__num_samples_batch)]
                data = frames['data'][0].tolist()
                return data, timestamps
            else:
                raise Exception
        except Exception:
            return [], None

    def start_daq(self, do_batch: bool=True, do_plot: bool=False, window_sec: float= 30., track_util: bool=False, folder_name: str="data") -> None:
        """Changing the state of the DAQ with starting it
        :param do_batch:    True for sending batches outside otherwise sample-wise
        :param do_plot:     True to plot the data in real-time
        :param window_sec:  Floating value with window length [in seconds] for live plotting
        :param track_util:  If true, the utilization (CPU / RAM) of the host computer will be tracked during recording session
        :param folder_name: String with folder name to save data in project folder
        :return: None
        """
        batchsize = self.get_number_samples_per_batch()
        self.__define_bytes_data(batchsize)
        path2data = get_path_to_project(new_folder=folder_name)

        func = self._thread_read_batch if do_batch else self._thread_read_frame
        self.__threads.register(func=self.__threads.lsl_stream_data, args=(0, 'data', func, 2, self.__sampling_rate))
        self.__threads.register(func=self.__threads.lsl_record_stream, args=(1, 'data', path2data))
        if track_util:
            self.__threads.register(func=self.__threads.lsl_stream_util, args=(2, 'util', 2.))
            self.__threads.register(func=self.__threads.lsl_record_stream, args=(3, 'util', path2data))
        if do_plot:
            self.__threads.register(func=self.__threads.lsl_plot_stream, args=(4 if track_util else 2, 'data', window_sec))

        self.__device.timeout = 2 / self.__sampling_rate
        self.__threads.start()
        self.__write_without_feedback(Commands.START_BATCH_DAQ if do_batch else Commands.START_SAMPLE_DAQ, 0)

    def stop_daq(self) -> None:
        """Changing the state of the DAQ with stopping it
        :return:            None
        """
        self.__threads.stop()
        self.__write_without_feedback(Commands.STOP_DAQ, 0)
        self.__device.timeout = self.__timeout_default

    def wait_daq(self, time_sec: float) -> None:
        """Waiting Routine incl. returning possible thread errors
        :param time_sec:    Float with time value for waiting
        :return:            None
        """
        self.__threads.wait_for_seconds(time_sec)

    def update_daq_sampling_rate(self, sampling_rate: float) -> None:
        """Updating the sampling rate of the DAQ
        :param sampling_rate:   Float with sampling rate [Hz]
        :return:                None
        """
        sampling_limits = [0, 10e3]
        if sampling_rate < sampling_limits[0]:
            raise ValueError(f"Sampling rate cannot be smaller than {sampling_limits[0]}")
        if sampling_rate > sampling_limits[1]:
            raise ValueError(f"Sampling rate cannot be greater than {sampling_limits[1]}")

        self.__sampling_rate = sampling_rate
        self.__write_without_feedback(Commands.UPDATE_DAQ, int(sampling_rate))

    def get_number_samples_per_batch(self) -> int:
        """Get number of samples per batch which are transmitted in one data package
        :return:    Integer with number of samples per batch
        """
        ret = self.__write_with_feedback(Commands.ASK_BATCH_DAQ, 0, 3)
        if ret[0] != Commands.ASK_BATCH_DAQ:
            raise ValueError(f"Get: {ret}")
        return int(ret[2])
