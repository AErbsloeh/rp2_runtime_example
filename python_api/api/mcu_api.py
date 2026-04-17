from enum import IntEnum
from logging import getLogger, Logger
from time import sleep
import numpy as np

from api.interface import (
    get_comport_name,
    InterfaceSerial
)
from api.lsl import ThreadLSL
from api.helper import (
    _convert_pin_state,
    _convert_system_state,
    _convert_rp2_temp_value,
    get_path_to_project,
    DataAcquisitionConfig,
    SystemState
)
    

class Commands(IntEnum):
    ECHO = 0x00
    RESET = 0x01
    GET_CHARAC_STATE = 0x02
    GET_CHARAC_DAQ = 0x03
    ENABLE_LED = 0x04
    DISABLE_LED = 0x05
    TOGGLE_LED = 0x06
    START_DAQ = 0x07
    STOP_DAQ = 0x08
    SET_PERIOD_DAQ = 0x09
    SET_BATCH_DAQ = 0x0A


class DeviceAPI:
    __logger: Logger
    __device: InterfaceSerial
    __threads: ThreadLSL
    __daq_config: DataAcquisitionConfig
    __timeout_default: float
    __usb_vid: int = 0x2E8A
    __last_idx: int = 255
    __num_package_loss: int = 0
    # PID of RP2350 = 0x0009 and RP2040 = 0x000A
    _pin_names: list[str] = ['LED_USER']
    _state_names: list[str] = ["ERROR", "RESET", "INIT", "IDLE", "TEST", "DAQ"]

    def __init__(self, com_name: str="AUTOCOM", timeout: float=1.) -> None:
        """Init. of the device with name and baudrate of the device
        :param com_name:    String with the serial port name of the used device
        :param timeout:     Floating value with timeout for the communication [Default, not during DAQ]
        """
        self.__logger = getLogger(__name__)
        self.__threads = ThreadLSL()
        self.__timeout_default = timeout
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

    def __write_with_feedback(self, head: Commands, data: int=0, size: int=0) -> bytes:
        data = self.__device.write_wfb(
            data=self.__device.convert(head, data),
            size=size
        )
        if data[0] != head:
            raise ValueError(f"Get: {data}")
        return data[1:]

    def __write_without_feedback(self, head: Commands, data: int=0) -> None:
        self.__device.write(
            data=self.__device.convert(head, data),
        )

    @staticmethod
    def _bytes_to_int(data: bytes, signed: bool=False) -> int:
        return int.from_bytes(data, byteorder='little', signed=signed)

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
        return self.get_state().system == "DAQ" and self.__threads.is_alive

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
        self.__write_without_feedback(Commands.RESET)
        sleep(4)

    def echo(self, data: str) -> str:
        """Sending some characters to the device and returning the result
        :param data:    String with the data to be sent
        :return:        String with returned data from DAQ
        """
        do_padding = len(data) % self.__device.num_bytes == 1
        val = bytes()
        for chunk in self.__device.serialize_string(data, do_padding):
            val += self.__write_with_feedback(Commands.ECHO, chunk)
        return self.__device.deserialize_string(val, do_padding)

    @property
    def _package_system_state(self) -> np.dtype:
        return np.dtype([
            ('state', '<u2'),
            ('clock', '<u2'),
            ('pins', '<u2'),
            ('temp', '<u2'),
            ('major', 'u1'),
            ('minor', 'u1'),
            ('runtime', '<u8')
        ])

    def get_state(self) -> SystemState:
        """Returning the state of the system
        :return:    Class SystemState with information about pin state, system state and actual runtime of the system
        """
        ret = self.__write_with_feedback(Commands.GET_CHARAC_STATE, size=19)
        frame = np.frombuffer(ret, dtype=self._package_system_state)[0]
        return SystemState(
            pins=_convert_pin_state(int(frame['pins']), self._pin_names),
            system=_convert_system_state(int(frame['state']), self._state_names),
            runtime=float(1e-6 * frame['runtime']),
            clock=10 * int(frame['clock']),
            firmware=f"{frame['major']}.{frame['minor']}",
            temp=_convert_rp2_temp_value(int(frame['temp']))
        )

    def enable_led(self) -> None:
        """Changing the state of the LED with enabling it
        :return:        None
        """
        self.__write_without_feedback(Commands.ENABLE_LED)

    def disable_led(self) -> None:
        """Changing the state of the LED with disabling it
        :return:        None
        """
        self.__write_without_feedback(Commands.DISABLE_LED)

    def toggle_led(self) -> None:
        """Changing the state of the LED with toggling it
        :return:        None
        """
        self.__write_without_feedback(Commands.TOGGLE_LED)

    def _update_daq_sampling_rate(self, sampling_rate: float) -> None:
        """Updating the sampling rate of the DAQ
        :param sampling_rate:   Float with sampling rate [Hz]
        :return:                None
        """
        sampling_limits = [0, 10e3]
        if not sampling_limits[0] < sampling_rate < sampling_limits[1]:
            raise ValueError(f"Sampling rate cannot be smaller than [{sampling_limits[0], sampling_limits[1]}] Hz")
        self.__write_without_feedback(Commands.SET_PERIOD_DAQ, int(sampling_rate))

    def _enable_batch_daq(self, use_batches: bool=True) -> None:
        """Enabling or disabling the batch transmission mode of the DAQ
        :param use_batches:     Boolean with True for enabling batch mode otherwise sample-wise
        :return:                None
        """
        self.__write_without_feedback(Commands.SET_BATCH_DAQ, int(use_batches))

    def _check_package_loss(self, new_idx: int) -> None:
        if 1 < new_idx - self.__last_idx < 255:
            self.__num_package_loss += 1
            self.__logger.debug(f"Package loss detected: {self.__num_package_loss}")
        self.__last_idx = new_idx

    @property
    def _package_daq_config(self) -> np.dtype:
        return np.dtype([
            ('head', 'u1'),
            ('tail', 'u1'),
            ('signed', 'u1'),
            ('mode', 'u1'),
            ('num_channels', '<u2'),
            ('num_samples', '<u2'),
            ('num_bytes', '<u2'),
            ('bytes_sample', 'u1'),
            ('period', '<i8')
        ])

    def get_daq_characteristics(self) -> DataAcquisitionConfig:
        """Get number of channels for acquiring data
        :return:    Integer with number of DAQ channels
        """
        ret = self.__write_with_feedback(Commands.GET_CHARAC_DAQ, size=20)
        frame = np.frombuffer(ret, dtype=self._package_daq_config)[0]
        return DataAcquisitionConfig(
            head_cmd=int(frame['head']),
            tail_cmd=int(frame['tail']),
            num_channels=int(frame['num_channels']),
            num_samples=int(frame['num_samples']),
            num_bytes_total=int(frame['num_bytes']),
            send_batch=bool(frame['mode']),
            bytes_sample=int(frame['bytes_sample']),
            sampling_rate=float(-1e6 / frame['period']),
            is_signed=bool(frame['signed'])
        )

    @property
    def _package_daq_sample(self) -> np.dtype:
        return np.dtype([
            ('head', 'u1'),
            ('index', 'u1'),
            ('timestamp', '<u8'),
            ('data', self.__daq_config.dtype_sample, self.__daq_config.data_shape),
            ('tail', 'u1')
        ])

    def _thread_read_frame(self) -> tuple[list, float]:
        try:
            buffer = self.__device.read(self.__daq_config.num_bytes_total)
            if not buffer:
                raise Exception
            frames = np.frombuffer(buffer, dtype=self._package_daq_sample)[0]
            mask = (frames['head'], frames['tail']) == (self.__daq_config.head_cmd, self.__daq_config.tail_cmd)
            if mask:
                self._check_package_loss(int(frames['index']))
                timestamps = 1e-6 * float(frames['timestamp'])
                data = frames['data'].tolist()
                return data, timestamps
            else:
                raise Exception
        except Exception:
            return [], None

    @property
    def _thread_batch_datatype(self) -> np.dtype:
        return np.dtype([
            ('head', 'u1'),
            ('index', 'u1'),
            ('timestamp', '<u8', (2,)),
            ('data', self.__daq_config.dtype_sample, self.__daq_config.data_shape),
            ('tail', 'u1')
        ])

    def _thread_read_batch(self) -> tuple[list[list], list[float]]:
        try:
            buffer = self.__device.read(self.__daq_config.num_bytes_total)
            if not buffer:
                raise Exception
            frames = np.frombuffer(buffer, dtype=self._thread_batch_datatype)[0]
            mask = (frames['head'], frames['tail']) == (self.__daq_config.head_cmd, self.__daq_config.tail_cmd)
            if mask:
                self._check_package_loss(int(frames['index']))
                dt = (frames['timestamp'][1] - frames['timestamp'][0]) / (self.__daq_config.num_samples-1)
                timestamps = [float(1e-6 * (frames['timestamp'][0] + dt*idx)) for idx in range(self.__daq_config.num_samples)]
                data = frames['data'].tolist()
                return data, timestamps
            else:
                raise Exception
        except Exception:
            return [], None

    def start_daq(self, sampling_rate: float, do_batch: bool=True, do_plot: bool=False, window_sec: float= 30., track_util: bool=False, folder_name: str="data") -> None:
        """Changing the state of the DAQ with starting it
        :param sampling_rate:   Float with sampling rate [Hz]
        :param do_batch:        True for sending batches outside otherwise sample-wise
        :param do_plot:         True to plot the data in real-time
        :param window_sec:      Floating value with window length [in seconds] for live plotting
        :param track_util:      If true, the utilization (CPU / RAM) of the host computer will be tracked during recording session
        :param folder_name:     String with folder name to save data in project folder
        :return:                None
        """
        self.__num_package_loss = 0
        self._enable_batch_daq(do_batch)
        self._update_daq_sampling_rate(sampling_rate)
        self.__daq_config = self.get_daq_characteristics()
        path2data = get_path_to_project(new_folder=folder_name)

        func = self._thread_read_batch if do_batch else self._thread_read_frame
        self.__threads.register(func=self.__threads.lsl_stream_data, args=(0, 'data', func, 2, self.__daq_config.sampling_rate))
        self.__threads.register(func=self.__threads.lsl_record_stream, args=(1, 'data', path2data))
        if track_util:
            self.__threads.register(func=self.__threads.lsl_stream_util, args=(2, 'util', 2.))
            self.__threads.register(func=self.__threads.lsl_record_stream, args=(3, 'util', path2data))
        if do_plot:
            self.__threads.register(func=self.__threads.lsl_plot_stream, args=(4 if track_util else 2, 'data', window_sec))

        self.__device.timeout = 2 / self.__daq_config.sampling_rate
        self.__threads.start()
        self.__write_without_feedback(Commands.START_DAQ)

    def stop_daq(self) -> None:
        """Changing the state of the DAQ with stopping it
        :return:            None
        """
        self.__write_without_feedback(Commands.STOP_DAQ)
        sleep(0.5)
        self.__device.timeout = self.__timeout_default
        while self.__threads.is_alive:
            self.__threads.stop()
            sleep(0.5)
        self.__device.empty_buffer()
        if self.__num_package_loss > 0:
            self.__logger.info(f"Number of package losses: {self.__num_package_loss}")

    def wait_daq(self, time_sec: float) -> None:
        """Waiting Routine incl. returning possible thread errors
        :param time_sec:    Float with time value for waiting
        :return:            None
        """
        self.__threads.wait_for_seconds(time_sec)
