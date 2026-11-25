from .definitions import Commands
from .src._basic_api import BasicAPI
from .src._helper import (
    DataAcquisitionConfig,
    get_path_to_project,
)


class DeviceAPI(BasicAPI):
    has_fpga: bool = False
    __daq_config: DataAcquisitionConfig

    def process_data(self, value: int) -> int | None:
        """Sending one signed integer to the processing function of the device (`run_sequential` in the firmware)
        The value must fit into the data bytes of one transmission (signed 24 bit: -8388608 ... 8388607)
        :param value:   Integer value to be processed by the device
        :return:        Integer value with the result from the device or None if the device sends no response
        """
        num_bytes = self._num_bytes_data
        raw = int(value).to_bytes(num_bytes, byteorder="big", signed=True)
        answer = self._device.write_wfb(
            data=self._device.convert(Commands.PROCESS_DATA, int.from_bytes(raw, byteorder="big"))
        )
        if len(answer) == 0:
            return None
        if len(answer) != self.total_num_bytes or answer[0] != Commands.PROCESS_DATA:
            raise ValueError(f"Get: {answer.hex(' ')}")
        return int.from_bytes(answer[1:], byteorder="big", signed=True)

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
        :param do_record:       True to record the LSL stream to a HDF5 file
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
            path2data = get_path_to_project(new_folder=folder_name)
            self._threads.register(
                func=self._threads.lsl_record_stream,
                args=(stream_idx, [output_stream, *stream_names[1:], "util"], path2data),
            )
            stream_idx += 1

        self._device.timeout = 2 / min(config.sampling_rate for config in daq_configs)
        self._threads.start()
        self._write_without_feedback(Commands.START_DAQ)

        if do_plot:
            if self._plot_config is None or not self._plot_config.channel_configs:
                raise RuntimeError("LivePlotter is not configured!")
            self._threads.lsl_plot_stream(plot_config=self._plot_config, window_length=window_sec)

    def set_liveplotter_config_manually(self, do_process: bool = True) -> None:
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
