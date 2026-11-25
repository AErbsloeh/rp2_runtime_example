import json
from dataclasses import replace
from logging import Logger, getLogger
from pathlib import Path
from time import monotonic, sleep

import numpy as np

from ..definitions import Commands
from ._helper import (
    DaqStreamDescription,
    DataAcquisitionConfig,
    SystemState,
    build_crc_excluding_endframe,
    convert_pin_state,
    convert_rp2_temp_value,
    convert_system_state,
    get_path_to_project,
)
from ._interface_serial import InterfaceSerial, get_comport_name
from ._interface_wifi import InterfaceWifi
from ._live_visualize import ChannelConfig, LivePlotterChannelConfig
from ._lsl import ThreadLSL


class BasicAPI:
    """Generic base of the device API: connection (USB or WiFi), RPC protocol, system state, LED and the reading
    and decoding of the DAQ packets. It does not depend on the application, the application-specific part
    (LSL threads, processing, plotting) is in DeviceAPI"""

    _logger: Logger
    _num_bytes_head: int = 1
    _num_bytes_data: int = 3
    _device: InterfaceSerial | InterfaceWifi
    _daq_streams: dict[int, DataAcquisitionConfig]
    _daq_dtypes: dict[int, np.dtype]
    _rx_buffer: bytearray = bytearray()
    _num_skipped_bytes: int = 0
    _timeout_default: float
    _usb_vid: int = 0x2E8A
    _last_idx: dict[int, int]
    _num_package_loss: int = 0
    _num_daq_tasks: int = 0
    _threads: ThreadLSL
    _plot_config: LivePlotterChannelConfig | None = None
    _pin_names: list[str] = ["LED_USER"]
    _state_names: list[str] = ["ERROR", "RESET", "INIT", "IDLE", "TEST", "DAQ"]

    def __init__(
        self,
        com_name: str = "AUTOCOM",
        timeout: float = 0.1,
        transport: str = "usb",
        host: str = "",
        port: int = 4242,
    ) -> None:
        """Interface class for handling with a custom DAQ device
        :param com_name:    String with the serial port name of the used device
        :param timeout:     Floating value with timeout for the communication [Default, not during DAQ]
        :param transport:   String with transport backend, either "usb" or "wifi"
        :param host:        String with IP address or hostname for WiFi transport
        :param port:        Integer with TCP port for WiFi transport
        """
        self._logger = getLogger(__name__)
        self._timeout_default = timeout

        transport = transport.lower()
        if transport == "usb":
            self._device = InterfaceSerial(
                com_name=com_name if com_name != "AUTOCOM" else get_comport_name(usb_vid=self._usb_vid),
                baud=230400,
                num_bytes_head=self._num_bytes_head,
                num_bytes_data=self._num_bytes_data,
                timeout=self._timeout_default,
            )
        elif transport == "wifi":
            if host == "":
                raise ValueError("host must be set when using WiFi transport")
            self._device = InterfaceWifi(
                host=host,
                port=port,
                num_bytes_head=self._num_bytes_head,
                num_bytes_data=self._num_bytes_data,
                timeout=self._timeout_default,
            )
        else:
            raise ValueError(f"Unsupported transport: {transport}")

        self._threads = ThreadLSL()
        self._daq_streams = dict()
        self._daq_dtypes = dict()
        self._last_idx = dict()
        if self.is_com_port_active:
            self._device.close()
        self._device.open()

    def extend_pins_list(self, new_pins: list[str]) -> None:
        for pin in new_pins:
            if pin not in self._pin_names:
                self._pin_names.append(pin)

    def extend_states_list(self, new_states: list[str]) -> None:
        for state in new_states:
            if state not in self._state_names:
                self._state_names.append(state)

    def _write_with_feedback(self, head: Commands, data: int = 0, size: int = 0) -> bytes:
        data = self._device.write_wfb(data=self._device.convert(head, data), size=size)
        if data[0] != head:
            raise ValueError(f"Get: {data.hex(' ')}")
        return data[1:]

    def _write_without_feedback(self, head: Commands, data: int = 0) -> None:
        self._device.write(
            data=self._device.convert(head, data),
        )

    def _write_bytes(self, data: bytes, size: int = 0) -> bytes:
        ret = self._device.write_wfb(data=data, size=size)
        return ret

    @staticmethod
    def _bytes_to_int(data: bytes, signed: bool = False) -> int:
        return int.from_bytes(data, byteorder="little", signed=signed)

    @property
    def total_num_bytes(self) -> int:
        """Returning the total number of bytes for each transmission"""
        return self._device.total_num_bytes

    @property
    def is_com_port_active(self) -> bool:
        """Boolean for checking if serial communication is open and used"""
        return self._device.is_open

    def open(self) -> None:
        """Opening the serial communication between API and device"""
        self._device.open()

    def close(self) -> None:
        """Closing the serial communication between API and device"""
        self._device.close()

    def do_reset(self) -> None:
        """Performing a Software Reset on the Platform (a running DAQ is stopped before)"""
        if self._threads.is_alive:
            self._threads.stop()
            self.stop_daq()
        self._write_without_feedback(Commands.RESET)
        self.close()
        if isinstance(self._device, InterfaceSerial):
            self._wait_for_usb_device()
        else:
            sleep(2.0)

    def _is_usb_device_available(self) -> bool:
        try:
            get_comport_name(usb_vid=self._usb_vid)
        except ConnectionError:
            return False
        return True

    def _wait_for_usb_device(self, timeout: float = 15.0, timeout_gone: float = 3.0) -> None:
        start = monotonic()
        while monotonic() - start < timeout_gone and self._is_usb_device_available():
            sleep(0.1)
        start = monotonic()
        while monotonic() - start < timeout:
            if self._is_usb_device_available():
                sleep(0.5)
                return
            sleep(0.2)
        raise ConnectionError(f"USB device did not return within {timeout} s after reset")

    def echo(self, data: str) -> str:
        """Sending some characters to the device and returning the result
        :param data:    String with the data to be sent
        :return:        String with returned data from DAQ
        """
        raw_len = len(data.encode("utf-8"))
        chunks = self._device.serialize_string(data)
        val = b"".join(self._write_with_feedback(Commands.ECHO, chunk) for chunk in chunks)
        return self._device.deserialize_string(val, raw_len)

    def echo_value(self, value: int) -> int:
        """Sending one signed integer to the device and returning the echoed value (loopback test with numbers)
        The value must fit into the data bytes of one transmission (signed 24 bit: -8388608 ... 8388607)
        :param value:   Integer value to be sent
        :return:        Integer value returned from the device
        """
        num_bytes = self._num_bytes_data
        raw = int(value).to_bytes(num_bytes, byteorder="big", signed=True)
        ret = self._write_with_feedback(Commands.ECHO, int.from_bytes(raw, byteorder="big"))
        return int.from_bytes(ret[:num_bytes], byteorder="big", signed=True)

    @property
    def _package_system_state(self) -> np.dtype:
        return np.dtype(
            [
                ("state", "<u2"),
                ("clock", "<u2"),
                ("pins", "<u2"),
                ("temp", "<u2"),
                ("major", "u1"),
                ("minor", "u1"),
                ("runtime", "<u8"),
            ]
        )

    def get_state(self) -> SystemState:
        """Returning the state of the system
        :return:    Class SystemState with information about pin state, system state and actual runtime of the system
        """
        ret = self._write_with_feedback(Commands.GET_CHARAC_STATE, size=19)
        frame = np.frombuffer(ret, dtype=self._package_system_state)[0]
        return SystemState(
            pins=convert_pin_state(int(frame["pins"]), self._pin_names),
            system=convert_system_state(int(frame["state"]), self._state_names),
            runtime=float(1e-6 * frame["runtime"]),
            clock=10 * int(frame["clock"]),
            firmware=f"{frame['major']}.{frame['minor']}",
            temp=convert_rp2_temp_value(int(frame["temp"])),
        )

    def enable_led(self) -> None:
        """Changing the state of the LED with enabling it
        :return:        None
        """
        self._write_without_feedback(Commands.ENABLE_LED)

    def disable_led(self) -> None:
        """Changing the state of the LED with disabling it
        :return:        None
        """
        self._write_without_feedback(Commands.DISABLE_LED)

    def toggle_led(self) -> None:
        """Changing the state of the LED with toggling it
        :return:        None
        """
        self._write_without_feedback(Commands.TOGGLE_LED)

    def _update_daq_sampling_rate(self, sampling_rate: float) -> None:
        """Updating the sampling rate of the DAQ
        :param sampling_rate:   Float with sampling rate [Hz]
        :return:                None
        """
        sampling_limits = [0, 10e3]
        if not sampling_limits[0] < sampling_rate < sampling_limits[1]:
            raise ValueError(
                f"Sampling rate cannot be smaller than [{sampling_limits[0], sampling_limits[1]}] Hz"
            )
        self._write_without_feedback(Commands.SET_PERIOD_DAQ, int(sampling_rate))

    def _enable_batch_daq(self, process_mode: int = 0) -> None:
        """Enabling or disabling the batch transmission mode of the DAQ
        :param process_mode:    Integer for selecting the processing mode (0: sample, 1: batch, 2: ping-pong batch)
        :return:                None
        """
        if process_mode not in range(0, 3):
            raise ValueError("Not the right mode")
        self._write_without_feedback(Commands.SET_BATCH_DAQ, data=process_mode)

    def _check_package_loss(self, head_cmd: int, new_idx: int) -> None:
        """Checking the (8-bit, wrapping) packet index of one stream for missing packets"""
        last_idx = self._last_idx.get(head_cmd)
        if last_idx is not None:
            num_lost = (new_idx - last_idx - 1) % 256
            if num_lost:
                self._num_package_loss += num_lost
                self._logger.debug(
                    f"Package loss detected on stream 0x{head_cmd:02X}: {self._num_package_loss}"
                )
        self._last_idx[head_cmd] = new_idx

    def _check_crc(self, packet: bytes, received_crc: int) -> bool:
        """Verify CRC of a received packet. If failed program will be stopped"""
        calculated_crc = build_crc_excluding_endframe(packet)
        state = calculated_crc == received_crc
        if not state:
            raise RuntimeError("CRC check failed.")
        return state

    def _get_number_daq_tasks(self) -> int:
        ret = self._write_with_feedback(Commands.GET_NUMBER_DAQ, data=0, size=2)
        self._num_daq_tasks = self._bytes_to_int(ret, signed=False)
        return self._num_daq_tasks

    @property
    def _package_daq_config(self) -> np.dtype:
        return np.dtype(
            [
                ("head", "u1"),
                ("tail", "u1"),
                ("signed", "u1"),
                ("mode", "u1"),
                ("num_channels", "<u2"),
                ("num_samples", "<u2"),
                ("num_bytes", "<u2"),
                ("bytes_sample", "u1"),
                ("period", "<u8"),
            ]
        )

    def get_daq_characteristics(self, sel_daq_tasks: int = 0) -> DataAcquisitionConfig:
        """Get number of channels for acquiring data
        :param sel_daq_tasks:   Integer with number of task config to acquire
        :return:                Integer with number of DAQ channels
        """
        if sel_daq_tasks not in range(self._num_daq_tasks):
            raise ValueError(f"Integer with selected DAQ task is out-of-range [0, {self._num_daq_tasks})")
        ret = self._write_with_feedback(Commands.GET_CHARAC_DAQ, data=sel_daq_tasks, size=20)
        frame = np.frombuffer(ret, dtype=self._package_daq_config)[0]
        return DataAcquisitionConfig(
            head_cmd=int(frame["head"]),
            tail_cmd=int(frame["tail"]),
            num_channels=int(frame["num_channels"]),
            num_samples=int(frame["num_samples"]),
            num_bytes_total=int(frame["num_bytes"]),
            send_batch=bool(frame["mode"]),
            bytes_sample=int(frame["bytes_sample"]),
            sampling_rate=float(1e6 / frame["period"]),
            is_signed=bool(frame["signed"]),
        )

    @staticmethod
    def _build_package_dtype(config: DataAcquisitionConfig) -> np.dtype:
        """Building the numpy datatype of one DAQ packet (sample or batch mode)"""
        num_values = config.num_channels * config.num_samples
        return np.dtype(
            [
                ("head", "u1"),
                ("index", "u1"),
                ("timestamp", "<u8", (2,) if config.send_batch else (1,)),
                ("data", config.dtype_sample, (num_values,)),
                ("crc", "<u2", (1,) if config.has_crc else (0,)),
                ("tail", "u1"),
            ]
        )

    def _register_daq_streams(self, configs: list[DataAcquisitionConfig]) -> None:
        """Registering all DAQ streams, which can be received on the device interface (identified by head byte)"""
        self._daq_streams = dict()
        self._daq_dtypes = dict()
        for config in configs:
            if config.head_cmd in self._daq_streams:
                raise ValueError(
                    f"DAQ streams must have unique head bytes (0x{config.head_cmd:02X} is used twice)"
                )
            dtype = self._build_package_dtype(config)
            if dtype.itemsize != config.num_bytes_total:
                raise ValueError(
                    f"DAQ stream 0x{config.head_cmd:02X}: packet layout ({dtype.itemsize} bytes) does not "
                    f"match reported size ({config.num_bytes_total} bytes)"
                )
            self._daq_streams[config.head_cmd] = config
            self._daq_dtypes[config.head_cmd] = dtype
        self._rx_buffer = bytearray()
        self._last_idx = dict()
        self._num_skipped_bytes = 0

    def _fill_rx_buffer(self, num_bytes: int) -> bool:
        """Reading from the device until the RX buffer contains at least num_bytes (False on timeout)"""
        while len(self._rx_buffer) < num_bytes:
            chunk = self._device.read(num_bytes - len(self._rx_buffer))
            if not chunk:
                return False
            self._rx_buffer += chunk
        return True

    def _is_valid_packet(self, config: DataAcquisitionConfig, packet: bytes) -> bool:
        if packet[-1] != config.tail_cmd:
            return False
        if config.has_crc:
            received_crc = int.from_bytes(packet[-3:-1], byteorder="little")
            return build_crc_excluding_endframe(packet) == received_crc
        return True

    def _read_next_packet(self) -> tuple[DataAcquisitionConfig, bytes] | None:
        """Reading the next valid packet of any registered DAQ stream from the device.
        Packets of different streams can arrive in any order and have different lengths. The head byte selects
        the stream config (and thus the length), tail byte + CRC validate the candidate. On a mismatch, only a
        single byte is dropped, so the parser re-synchronises on the next real packet start.
        :return:    Tuple with (stream config, raw packet bytes) or None on timeout
        """
        buffer = self._rx_buffer
        max_packet_size = max(config.num_bytes_total for config in self._daq_streams.values())
        num_skipped = 0
        while num_skipped <= 2 * max_packet_size:
            if not self._fill_rx_buffer(1):
                return None
            config = self._daq_streams.get(buffer[0])
            if config is not None:
                if not self._fill_rx_buffer(config.num_bytes_total):
                    return None
                packet = bytes(buffer[: config.num_bytes_total])
                if self._is_valid_packet(config, packet):
                    del buffer[: config.num_bytes_total]
                    return config, packet
            del buffer[0]
            num_skipped += 1
            self._num_skipped_bytes += 1
        return None

    def _thread_read_packet(self) -> tuple[int | None, list, float | list[float] | None]:
        """Reading and decoding the next DAQ packet of any registered stream
        :return:    Tuple with (head_cmd, data, timestamp). Sample mode: data=list[channels], timestamp=float.
                    Batch mode: data=list[samples][channels], timestamp=list[float]. (None, [], None) if no data.
        """
        try:
            ret = self._read_next_packet()
            if ret is None:
                return None, [], None
            config, packet = ret
            frame = np.frombuffer(packet, dtype=self._daq_dtypes[config.head_cmd])[0]
            self._check_package_loss(config.head_cmd, int(frame["index"]))

            if config.send_batch:
                data = frame["data"].reshape(config.num_samples, config.num_channels).tolist()
                t_first, t_last = (int(val) for val in frame["timestamp"])
                dt = (t_last - t_first) / max(config.num_samples - 1, 1)
                timestamps = [1e-6 * (t_first + dt * idx) for idx in range(config.num_samples)]
                return config.head_cmd, data, timestamps
            else:
                return config.head_cmd, frame["data"].tolist(), 1e-6 * float(frame["timestamp"][0])
        except Exception as e:
            self._logger.debug(f"Failed to read DAQ packet: {e}")
            return None, [], None

    def _drain_input_buffer(self, max_tries: int = 25, wait_sec: float = 0.2) -> None:
        self._device.empty_buffer()
        for idx in range(max_tries):
            sleep(wait_sec)
            if self._device.empty_buffer() == 0:
                return
            if idx == max_tries // 2:
                self._write_without_feedback(Commands.STOP_DAQ)
        self._logger.warning("Device still sends data after stopping the DAQ")

    @staticmethod
    def _build_stream_descriptions(
        stream_names: list[str], configs: list[DataAcquisitionConfig]
    ) -> list[DaqStreamDescription]:
        """Building the descriptions of the LSL streams (default labels, units and layout) for the DAQ streams
        :param stream_names:    List with the LSL name of each DAQ stream
        :param configs:         List with the configuration of each DAQ stream
        :return:                List with the description of each stream
        """
        return [
            DaqStreamDescription(
                name=name,
                head_cmd=config.head_cmd,
                sampling_rate=config.sampling_rate,
                channel_labels=[f"CH{idx}" for idx in range(config.num_channels)],
                channel_units=["LSB" for _ in range(config.num_channels)],
                channel_layout=[[idx] for idx in range(config.num_channels)],
                is_batch=config.send_batch,
            )
            for name, config in zip(stream_names, configs)
        ]

    def _prepare_daq(
        self, sampling_rate: float, stream_names: list[str] | None = None
    ) -> tuple[list[DataAcquisitionConfig], list[DaqStreamDescription]]:
        """Preparing the DAQ of the device: setting the sampling rate, reading the configuration of all DAQ streams,
        registering them for reading and building the description of the LSL streams (incl. the plot definitions)
        :param sampling_rate:   Float with sampling rate [Hz]
        :param stream_names:    List with LSL names for each DAQ stream of the device (in firmware order)
                                Default: ["data0", "data1", "data2", ...]
        :return:                Tuple with (1) configuration and (2) LSL description of each DAQ stream
        """
        self._num_package_loss = 0
        self._num_skipped_bytes = 0
        self._update_daq_sampling_rate(sampling_rate)
        num_daq_tasks = self._get_number_daq_tasks()
        if num_daq_tasks < 1:
            raise RuntimeError("Device does not report any DAQ stream")
        daq_configs = [self.get_daq_characteristics(idx) for idx in range(num_daq_tasks)]
        self._register_daq_streams(daq_configs)

        if stream_names is None:
            stream_names = ["data0"] + [f"data{idx}" for idx in range(1, num_daq_tasks)]
        if len(stream_names) != num_daq_tasks or len(set(stream_names)) != num_daq_tasks:
            raise ValueError(f"stream_names needs {num_daq_tasks} unique names, got {stream_names}")
        streams = [
            replace(
                stream,
                **dict(
                    zip(
                        ("channel_labels", "channel_units", "channel_layout"),
                        self._get_stream_metadata(
                            stream.name,
                            stream.channel_labels,
                            stream.channel_units,
                            stream.channel_layout,
                        ),
                    )
                ),
            )
            for stream in self._build_stream_descriptions(stream_names, daq_configs)
        ]
        self._logger.debug(f"DAQ streams: {streams}")
        return daq_configs, streams

    @property
    def is_daq_running(self) -> bool:
        """Returning if DAQ is still running"""
        return self._threads.is_alive

    def setup_live_plotter(self, plot_titles: list[list[str | None]]) -> None:
        """initialize the live plotter ( row * col).
        channels can be added afterwards with add_plot_channel()
        """
        self._plot_config = LivePlotterChannelConfig(plot_title=plot_titles, channel_configs=[])

    def add_plot_channel(
        self,
        lsl_stream_name: str,
        channel_name: str,
        channel_unit: str,
        curve_color: str,
        position: tuple[int, int],
        stream_channel_idx: int = 0,
    ) -> None:
        """add a channel to the live plotter"""
        if self._plot_config is None:
            raise RuntimeError("please set up setup_live_plotter(plot_titles=..) first!")

        self._plot_config.channel_configs.append(
            ChannelConfig(
                lsl_stream_name=lsl_stream_name,
                channel_name=channel_name,
                channel_unit=channel_unit,
                curve_color=curve_color,
                position=position,
                stream_channel_idx=stream_channel_idx,
            )
        )

    def _get_stream_metadata(
        self, stream_name: str, labels: list[str], units: list[str], layout: list[list[int]]
    ) -> tuple[list[str], list[str], list[list[int]]]:
        """Getting labels, units and layout of an LSL stream, channels defined in the plot configuration override the defaults
        :param stream_name: String with name of the LSL stream
        :param labels:      List with default labels of each channel
        :param units:       List with default units of each channel
        :param layout:      List with default layout of each channel
        :return:            Tuple with (1) labels, (2) units and (3) layout of each channel
        """
        labels, units, layout = list(labels), list(units), list(layout)
        if self._plot_config is None:
            return labels, units, layout
        for ch in self._plot_config.channel_configs:
            if ch.lsl_stream_name != stream_name:
                continue
            idx = ch.stream_channel_idx
            if not 0 <= idx < len(labels):
                raise ValueError(
                    f"Plot channel '{ch.channel_name}' uses channel {idx}, but stream '{stream_name}' has {len(labels)} channels"
                )
            labels[idx] = ch.channel_name
            units[idx] = ch.channel_unit
            layout[idx] = list(ch.position)
        return labels, units, layout

    def _store_c_reader_stats(self, stats) -> None:
        self._num_package_loss = stats.num_lost_packets
        self._num_skipped_bytes = stats.num_skipped_bytes
        if stats.num_push_errors > 0:
            self._logger.warning(f"Number of failed LSL pushes in the C reader: {stats.num_push_errors}")

    def stop_daq(self) -> None:
        """Changing the state of the DAQ with stopping it
        :return:            None
        """
        self._write_without_feedback(Commands.STOP_DAQ)
        sleep(0.5)
        self._device.timeout = self._timeout_default
        self._threads.stop()
        self._drain_input_buffer()
        if self._num_package_loss > 0:
            self._logger.info(f"Number of package losses: {self._num_package_loss}")
        if self._num_skipped_bytes > 0:
            self._logger.info(f"Number of skipped bytes (re-sync): {self._num_skipped_bytes}")

    def wait_daq(self, time_sec: float) -> None:
        """Waiting routine incl. returning possible thread errors
        :param time_sec:    Float with time value for waiting
        :return:            None
        """
        self._threads.wait_for_seconds(time_sec)

    def load_plotconfigs_from_json(self, file_path: str | Path) -> None:
        """Loading plot and channel layout configurations from a JSON file
        :param file_path:   Path to the JSON configuration file
        :return:            Dictionary containing the parsed plot configurations
        """
        path = Path(get_path_to_project(), file_path)

        with open(path, "r", encoding="utf-8") as json_file:
            raw_data = json.load(json_file)

        channel_configs = [
            ChannelConfig(
                lsl_stream_name=ch["lsl_stream_name"],
                channel_name=ch["channel_name"],
                channel_unit=ch["channel_unit"],
                curve_color=ch["curve_color"],
                position=tuple(ch["position"]),
                stream_channel_idx=ch.get("stream_channel_idx", 0),
            )
            for ch in raw_data["channel_configs"]
        ]

        self._plot_config = LivePlotterChannelConfig(
            plot_title=raw_data["plot_title"],
            channel_configs=channel_configs,
        )
        self._logger.info(f"Loaded plot configurations from {path}")

    def save_plotconfigs_to_json(
        self,
        file_path: str | Path,
    ) -> None:
        """Saving current plot and channel layout configurations to a JSON file
        :param file_path:       Path to the target JSON configuration file

        :return:                Dictionary containing the saved plot configurations
        """
        path = Path(get_path_to_project(), file_path)

        with open(path, "w", encoding="utf-8") as json_file:
            json.dump(
                self._plot_config, json_file, default=lambda o: o.__dict__, indent=4, ensure_ascii=False
            )
        self._logger.info(f"Saved plot configurations to {path}")
