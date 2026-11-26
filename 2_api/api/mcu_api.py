from pathlib import Path

import numpy as np

from .definitions import Commands
from .src._basic_api import BasicAPI
from .src._helper import (
    DataAcquisitionConfig,
    get_path_to_project,
)


DEFAULT_PLOT_CONFIG = Path("plot_config.json")


class DeviceAPI(BasicAPI):
    has_fpga: bool = False
    __daq_config: DataAcquisitionConfig

    def apply_data(self, data: bytes | bytearray | np.ndarray) -> None:
        """Writing data into the input block of the embedded processing (command APPLY_DATA, no response)
        Each transmission has a counter (first data byte, index of the transmission) and the next bytes of the block, so the
        device puts the bytes to the right position even if a transmission is lost. The bytes are in the order of the block (little
        endian values, channels and samples as in the DAQ streams). An array is sent as it is in memory (C order), so its shape and
        data type have to match the layout of `get_processing_layout`. Bytes which are not applied are 0, bytes behind the end of the
        block are dropped. The processing starts with `process_data`
        :param data:    Bytes or array with the content of the input block
        """
        if isinstance(data, np.ndarray):
            raw = np.ascontiguousarray(data.astype(data.dtype.newbyteorder("<"))).tobytes()
        else:
            raw = bytes(data)
        num_payload = self._num_bytes_data - 1
        num_frames = -(-len(raw) // num_payload)
        if num_frames > 256:
            raise ValueError(
                f"{len(raw)} bytes need {num_frames} transmissions, the counter allows 256 ({256 * num_payload} bytes)"
            )
        for counter in range(num_frames):
            chunk = raw[counter * num_payload : (counter + 1) * num_payload].ljust(num_payload, b"\x00")
            frame = bytes([counter]) + chunk
            self._write_without_feedback(Commands.APPLY_DATA, int.from_bytes(frame, byteorder="little"))

    def process_data(self, data: int | bytes | bytearray | np.ndarray | None = None) -> None:
        """Handing the input block of the device to the processing function (`run_sequential` in the firmware)
        The device sends no response, the result is part of the DAQ stream of the processing output (`data1`). Read it with
        `read_processing_results` (or from the LSL stream `data1` while the DAQ runs).
        The block is 0 after each call, so it has to be applied again for the next call
        :param data:    None to process the block of `apply_data`; an integer goes into the first value of the block (signed 24 bit,
                        cut to the bytes of one value of the input block, the rest is 0); bytes or an array are applied with
                        `apply_data` first
        """
        if isinstance(data, (int, np.integer)):
            raw = int(data).to_bytes(self._num_bytes_data, byteorder="little", signed=True)
            num_bytes_value = self.get_processing_layout().in_bytes_value
            fill = b"\xff" if data < 0 else b"\x00"
            self.apply_data(raw[:num_bytes_value].ljust(num_bytes_value, fill))
        elif data is not None:
            self.apply_data(data)
        self._write_without_feedback(Commands.PROCESS_DATA)

    def read_processing_results(self, stream_index: int = -1, timeout: float = 0.05) -> list[list[int]]:
        """Reading the results of the embedded processing from the DAQ stream of the processing output (`daq_out` in the firmware)
        The device sends the results block by block (e.g. 32 samples per packet), so the values of several `process_data` calls
        arrive together and later. The packets of other streams are dropped. It reads the device itself, so do not use it while the
        DAQ threads of `start_daq` read the device (in this case use the LSL stream of the DAQ stream, e.g. `data1`)
        :param stream_index:    Index of the DAQ stream with the results in firmware order (-1: last stream)
        :param timeout:         Time in seconds to wait for the next packet
        :return:                List with [samples][channels] of all packets received until the timeout (empty without data)
        """
        if not self._daq_streams:
            num_streams = self._get_number_daq_tasks()
            self._register_daq_streams([self.get_daq_characteristics(idx) for idx in range(num_streams)])
        head_result = list(self._daq_streams)[stream_index]
        timeout_before = self._device.timeout
        self._device.timeout = timeout
        results: list[list[int]] = []
        try:
            while True:
                head, data, _ = self._thread_read_packet()
                if head is None:
                    break
                if head == head_result:
                    results.extend(data if self._daq_streams[head].send_batch else [data])
        finally:
            self._device.timeout = timeout_before
        return results

    def start_daq(
        self,
        sampling_rate: float,
        window_sec: float = 30.0,
        do_plot: bool = True,
        do_record: bool = True,
        do_process: bool = True,
        folder_name: str = "runs",
        stream_names: list[str] | None = None,
        use_c_reader: bool = False,
    ) -> None:
        """Changing the state of the DAQ with starting it
        :param sampling_rate:   Float with sampling rate [Hz]
        :param do_plot:         True to plot the data in real-time
        :param do_record:       True to record all LSL streams (raw DAQ streams, processed stream and utilization) to a HDF5 file
        :param do_process:      True to publish the processed LSL stream (of the first DAQ stream)
        :param window_sec:      Floating value with window length [in seconds] for live plotting
        :param folder_name:     String with folder name to save data in project folder
        :param stream_names:    List with LSL names for each DAQ stream of the device (in firmware order)
                                Default: ["data0", "data1", "data2", ...]
        :param use_c_reader:    True to read the device, decode the packets and push them into LSL with the C reader
                                (c_reader, POSIX only, compiled on first use) instead of the Python reader thread
        :return:                None
        """
        daq_configs, streams = self._prepare_daq(sampling_rate, stream_names)
        stream_names = [stream.name for stream in streams]
        self.__daq_config = daq_configs[0]
        if do_plot:
            if self._ensure_plot_config(DEFAULT_PLOT_CONFIG, lambda: self._setup_default_plot_layout(do_process)):
                streams = self._build_streams(stream_names, daq_configs)
        default_filt = self._build_stream_descriptions(["filt"], daq_configs[:1])[0]
        filt_labels, filt_units, filt_layout = self._get_stream_metadata(
            "filt", default_filt.channel_labels, default_filt.channel_units, default_filt.channel_layout
        )

        stream_idx = 0
        source_stream = stream_names[0]
        output_stream = source_stream

        if do_record:
            self._threads.register(func=self._threads.lsl_stream_util, args=(stream_idx, "util"))
            stream_idx += 1

        require_system_consumers = do_process or do_record or do_plot
        if use_c_reader:
            self._threads.register(
                func=self._threads.lsl_stream_system_c,
                args=(stream_idx, streams, daq_configs, self._device.fileno()),
                kwargs={
                    "require_consumers": require_system_consumers,
                    "on_finished": self._store_c_reader_stats,
                },
            )
        else:
            self._threads.register(
                func=self._threads.lsl_stream_system,
                args=(
                    stream_idx,
                    streams,
                    self._thread_read_packet,
                ),
                kwargs={"require_consumers": require_system_consumers},
            )
        stream_idx += 1

        if do_process:
            output_stream = "filt"
            require_process_consumers = do_record or do_plot
            self._threads.register(
                func=self._threads.lsl_process_stream,
                args=(
                    stream_idx,
                    source_stream,
                    output_stream,
                    self.__daq_config.num_samples,
                    self.__process_stream_init,
                    self.__process_stream_func,
                ),
                kwargs={
                    "require_consumers": require_process_consumers,
                    "channel_labels": filt_labels,
                    "channel_units": filt_units,
                    "channel_layout": filt_layout,
                },
            )
            stream_idx += 1

        if do_record:
            record_names = [output_stream]
            if do_process:
                record_names.append(source_stream)
            record_names += [*stream_names[1:], "util"]
            path2data = get_path_to_project(new_folder=folder_name)
            self._threads.register(
                func=self._threads.lsl_record_stream,
                args=(stream_idx, record_names, path2data),
            )
            stream_idx += 1

        self._device.timeout = 2 / min(config.sampling_rate for config in daq_configs)
        self._threads.start()
        self._write_without_feedback(Commands.START_DAQ)

        if do_plot:
            self._threads.lsl_plot_stream(plot_config=self._plot_config, window_length=window_sec)

    def _setup_default_plot_layout(self, do_process: bool) -> None:
        self.setup_live_plotter(plot_titles=[["plot1", "plot2"], [None, "plot3"]])
        self.add_plot_channel(
            lsl_stream_name="data0",
            channel_name="CH0 Raw",
            channel_unit="LSB",
            curve_color="r",
            position=(0, 0),
            stream_channel_idx=0,
        )
        self.add_plot_channel(
            lsl_stream_name="data0",
            channel_name="CH0 Temp",
            channel_unit="LSB",
            curve_color="b",
            position=(1, 1),
            stream_channel_idx=1,
        )
        self.add_plot_channel(
            lsl_stream_name="data1",
            channel_name="CH1 Raw",
            channel_unit="LSB",
            curve_color="g",
            position=(0, 1),
            stream_channel_idx=0,
        )
        if do_process:
            self.add_plot_channel(
                lsl_stream_name="filt",
                channel_name="CH0 Filtered",
                channel_unit="LSB",
                curve_color="g",
                position=(0, 0),
                stream_channel_idx=0,
            )

    def __process_stream_init(self, sampling_rate: float) -> None:
        pass

    def __process_stream_func(self, data_in: list) -> list:
        data_out = list()
        for data in data_in:
            dat = [(65535 - data[0]), data[1]]
            data_out.append(dat)
        return data_out
