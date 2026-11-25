import sys
from datetime import datetime
from logging import Logger, getLogger
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Lock, Thread
from time import monotonic, sleep

import numpy as np
import pylsl
from h5py import File
from psutil import cpu_percent, virtual_memory
from pylsl import StreamInfo, StreamInlet, StreamOutlet, proc_threadsafe, resolve_bypred
from pyqtgraph.Qt import QtWidgets
from tqdm import tqdm

from api.data_api import DataAPI, StreamRecording
from api.src._helper import DaqStreamDescription
from api.src._live_visualize import LivePlotter, LivePlotterChannelConfig


def numpy_to_lsl_format(dtype: type | np.dtype) -> int:
    """Getting the LSL channel format (cf_*) that fits to a NumPy data type
    :param dtype:   NumPy data type (e.g. np.float32 or np.dtype("int16"))
    :return:        Integer with LSL datatype format cf_*
    """
    if np.issubdtype(dtype, np.float32):
        return pylsl.cf_float32
    if np.issubdtype(dtype, np.floating):
        return pylsl.cf_double64
    if np.issubdtype(dtype, np.int8):
        return pylsl.cf_int8
    if np.issubdtype(dtype, np.int16):
        return pylsl.cf_int16
    if np.issubdtype(dtype, np.int32):
        return pylsl.cf_int32
    if np.issubdtype(dtype, np.integer):
        return pylsl.cf_int64
    raise ValueError(f"No LSL datatype format available for {dtype}")


class RingBuffer:
    def __init__(self, size) -> None:
        self._size = size
        self._data0 = np.zeros(shape=(size, 2), dtype=float)
        self._index = 0
        self._is_full = False

    def _update_index(self) -> None:
        self._index += 1
        if self._index >= self._size:
            self._index = 0
            self._is_full = True

    def append(self, value: int) -> None:
        self._data0[self._index, 0] = self._index
        if self._is_full:
            self._data0[:-1, 1] = self._data0[1:, 1]
            self._data0[-1, 1] = value
        else:
            self._data0[self._index, 1] = value
        self._update_index()

    def append_with_timestamp(self, tb: float, value: int) -> None:
        if self._is_full:
            self._data0[:-1, 0] = self._data0[1:, 0]
            self._data0[:-1, 1] = self._data0[1:, 1]
            self._data0[-1, 0] = tb
            self._data0[-1, 1] = value
        else:
            self._data0[self._index :, 0] = tb
            self._data0[self._index :, 1] = value
        self._update_index()

    def get_data(self) -> np.ndarray:
        return self._data0


class ThreadLSL:
    _logger: Logger
    _event: Event
    _lock: Lock
    _thread: list[Thread]
    _exception: Queue
    _thread_active: list[int]
    _is_active: bool
    _num_missed: int = 0
    _plotter: LivePlotter | None = None

    def __init__(self) -> None:
        """Class for managing all threads for the LSL data processing
        :return:    None
        """
        self._logger = getLogger(__name__)
        self._event = Event()
        self._lock = Lock()
        self._exception = Queue()
        self._release_threads()

    @property
    def is_alive(self) -> bool:
        """Returning the state of all threads if they are alive"""
        if len(self._thread) == 0:
            return False
        else:
            return all([thread.is_alive() for thread in self._thread])

    @property
    def is_running(self) -> bool:
        """Returning the state of the thread handler if DAQ is running"""
        return self._event.is_set() and self._is_active

    @staticmethod
    def _get_h5_format(mode: int) -> str:
        """Formatting a mode into string"""
        match mode:
            case pylsl.cf_float32:
                return "float32"
            case pylsl.cf_double64:
                return "float64"
            case pylsl.cf_string:
                return "string"
            case pylsl.cf_int32:
                return "int32"
            case pylsl.cf_int16:
                return "int16"
            case pylsl.cf_int8:
                return "int8"
            case pylsl.cf_int64:
                return "int64"
            case _:
                raise ValueError("Unknown LSL datatype format")

    def register(self, func, args, kwargs: dict | None = None) -> None:
        """Registering a thread with a custom instruction
        :param func:    Function object for further processing in own thread
        :param args:    Arguments of the object for starting it
        :param kwargs:  Keyword arguments of the object for starting it
        :return:        None
        """
        if not len(self._thread):
            self._thread = [Thread(target=self._thread_watchdog_heartbeat, args=(), daemon=True)]
            self._logger.debug("Registering thread: heartbeat")
        self._thread.append(Thread(target=self._run_thread, args=(func, args, kwargs or {}), daemon=True))
        self._logger.debug(f"Registering thread: {func.__name__}")

    def _run_thread(self, func, args, kwargs: dict) -> None:
        try:
            func(*args, **kwargs)
        except Exception as e:
            self._logger.error(f"Thread {func.__name__} stopped with error: {e!r}")
            with self._lock:
                self._exception.put(e)

    def start(self) -> None:
        """Starting all threads including heartbeat watchdog in own thread
        :return:    None
        """
        self._thread_active = list()
        if len(self._thread) < 2:
            raise AssertionError("No threads registered")
        else:
            self._thread_active = [False for _ in self._thread[1:]]
            self._logger.debug("Enabled boolean array for all threads")
            self._num_missed = 0
            self._is_active = True
            self._event.set()
            for idx, p in enumerate(self._thread):
                p.start()
            self._logger.debug("Enabling all LSL threads")
            sleep(0.2)

    def stop(self) -> None:
        """Stopping all threads and waiting for shutdown all threads
        :return:        None
        """
        self._event.clear()
        for p in self._thread:
            p.join(timeout=30.0)
        self._release_threads()
        self._logger.debug("Disabling all LSL threads")

    @staticmethod
    def _to_str_list(values: list | str | None, length: int) -> list[str]:
        if not values:
            return [""] * length
        if isinstance(values, str):
            return [values] * length
        return [str(val) if val is not None else "" for val in values]

    @staticmethod
    def _get_number_stream_samples(sampling_rate: float) -> int:
        return int(sampling_rate / 50) if sampling_rate > 500.0 else 10

    def _check_exception(self) -> None:
        """Function for checking if any exception information is available from any thread and return it
        :return:    None
        """
        try:
            exc = self._exception.get_nowait()
        except Empty:
            pass
        else:
            raise exc

    def wait_for_seconds(self, wait_time_sec: float) -> None:
        """Waiting routine during start and stop of DAQ and returning errors if happens
        :param wait_time_sec:   Time to wait for a second
        :return:                None
        """
        sleep(1.0)
        for _ in tqdm(range(int(wait_time_sec))):
            if self._is_active:
                self._check_exception()
                sleep(1.0)
            else:
                raise RuntimeError(f"One thread is shutdown [{self._is_active}] - {self._thread_active}")
        self._check_exception()

    def _establish_lsl_outlet(
        self,
        idx: int,
        lsl_name: str,
        lsl_type: str,
        sampling_rate: float,
        channel_num: int,
        channel_labels: list[str],
        channel_units: list[str],
        channel_layout: list[list[int]] | list[str],
        channel_type: int = pylsl.cf_int16,
        check_for_consumers: bool = False,
    ) -> StreamOutlet:
        info = StreamInfo(
            name=lsl_name,
            type=lsl_type,
            channel_count=channel_num,
            nominal_srate=sampling_rate,
            channel_format=channel_type,
            source_id=f"{lsl_name}_uid",
        )
        info.set_channel_labels(channel_labels)
        if len(channel_layout) > 0 and all(isinstance(val, str) for val in channel_layout):
            info.set_channel_types([str(val) for val in channel_layout])
        else:
            info.set_channel_types(str(list(channel_layout)))
        info.set_channel_units(channel_units)

        outlet = StreamOutlet(info)
        self._logger.debug(f"LSL outlet ({lsl_name}, {idx}): {outlet.get_info()}")
        if check_for_consumers:
            if not outlet.wait_for_consumers(timeout=10.0):
                raise ConnectionError("no consumers available")
        self._logger.debug(f"LSL outlet ({lsl_name}, {idx}): running")
        return outlet

    def _establish_lsl_inlet(self, name: str, timeout: float = 10.0) -> StreamInlet:
        info = resolve_bypred(predicate=f"name='{name}'", minimum=1, timeout=timeout)
        if not info:
            raise ValueError(f"LSL stream with {name} not found")
        inlet = StreamInlet(
            info=info[0], max_buflen=60, max_chunklen=1024, recover=True, processing_flags=proc_threadsafe
        )
        self._logger.debug(f"LSL inlet ({name}): {info}")
        return inlet

    def _release_threads(self) -> None:
        self._thread = []
        self._thread_active = []
        self._is_active = False

    def _thread_watchdog_heartbeat(self) -> None:
        while self._event.is_set() and self._is_active:
            check_alive = all([thread.is_alive() for thread in self._thread[1:]])
            try:
                with self._lock:
                    self._logger.debug(f"LSL thread feedback: {self._thread_active}")
                    checker = all(self._thread_active)
                    for i, _ in enumerate(self._thread_active):
                        self._thread_active[i] = False
                if check_alive and checker:
                    self._num_missed = 0
                else:
                    self._num_missed += 1
                    self._logger.debug(f"Detected missed package: {self._num_missed}")
                if self._num_missed >= 5:
                    self._is_active = False
                    self._logger.debug("Disabling LSL threads")
            except Exception as e:
                with self._lock:
                    self._exception.put(e)
            sleep(2.0)

    def _thread_dummy(self, stim_idx: int) -> None:
        while self._event.is_set() and self._is_active:
            try:
                with self._lock:
                    self._thread_active[stim_idx] = True
                sleep(0.1)
            except Exception as e:
                with self._lock:
                    self._exception.put(e)
        with self._lock:
            self._thread_active[stim_idx] = False

    def lsl_stream_mock(
        self,
        stim_idx: int,
        name: str,
        channel_num: int = 2,
        sampling_rate: float = 200.0,
        format: int = pylsl.cf_int8,
    ) -> None:
        """Process for starting a Lab Streaming Layer (LSL) to mock the DAQ hardware with random data
        :param stim_idx:        Integer with array index to write into heartbeat feedback array
        :param name:            String with the name of the LSL stream (must match with a recording process)
        :param channel_num:     Channel number to start stream from
        :param sampling_rate:   Floating value with sampling rate in Hz
        :param format:          Defined datatype for saving data in the stream
        :return:                None
        """
        outlet = self._establish_lsl_outlet(
            idx=stim_idx,
            lsl_name=name,
            lsl_type="mock_daq",
            sampling_rate=sampling_rate,
            channel_units=["" for _ in range(channel_num)],
            channel_labels=[f"CH{idx}" for idx in range(channel_num)],
            channel_layout=[idx for idx in range(channel_num)],
            channel_num=channel_num,
            channel_type=pylsl.cf_int16,
            check_for_consumers=False,
        )

        is_int = format in [pylsl.cf_int8, pylsl.cf_int16, pylsl.cf_int32, pylsl.cf_int64]

        while self._event.is_set() and self._is_active:
            try:
                # Process data
                val = (
                    np.random.random(size=channel_num)
                    if not is_int
                    else np.random.randint(low=-(2**15), high=2**15, size=channel_num)
                )
                outlet.push_sample(
                    x=val.tolist(),
                    timestamp=0.0,
                    pushthrough=True,
                )
                # Heartbeat
                with self._lock:
                    self._thread_active[stim_idx] = outlet.have_consumers()
                sleep(1 / sampling_rate)
            except Exception as e:
                with self._lock:
                    self._exception.put(e)
        with self._lock:
            self._thread_active[stim_idx] = False

    def lsl_stream_file(
        self,
        stim_idx: int,
        name: str,
        path2data: str,
        name_type: str,
        file_index: int = -1,
        format: int = pylsl.cf_int16,
    ) -> None:
        """Process for starting a Lab Streaming Layer (LSL) to stream the file content into the DAQ system by mocking the DAQ hardware
        :param stim_idx:        Integer with array index to write into heartbeat feedback array
        :param name:            String with name of the LSL stream (must match with a recording process)
        :param path2data:       Path to folder with pre-recorded data files
        :param name_type:       Type name of loading the pre-recorded data from file
        :param file_index:      Number of the file in the folder to process
        :param format:          Defined datatype for saving data in the stream
        :return:                None
        """
        path0 = Path(path2data)
        if not path0.exists():
            raise AttributeError("File is not available")
        reader = DataAPI(path2data=path0)
        reader.select_file(file_index)
        data: StreamRecording = reader.get_data(name_type)

        outlet = self._establish_lsl_outlet(
            idx=stim_idx,
            lsl_name=name,
            lsl_type="mock_daq",
            channel_units=data.units,
            sampling_rate=float(data.sampling_rate),
            channel_labels=data.label,
            channel_layout=data.layout,
            channel_num=int(data.num_channels),
            channel_type=format,
            check_for_consumers=False,
        )

        while self._event.is_set() and self._is_active:
            try:
                for stime, sdata in zip(data.time, data.data.T):
                    if not self._event.is_set():
                        break
                    # Heartbeat
                    with self._lock:
                        self._thread_active[stim_idx] = outlet.have_consumers()
                    # Process data
                    outlet.push_sample(x=sdata.tolist(), timestamp=stime, pushthrough=True)
                    sleep(1 / data.sampling_rate)
            except Exception as e:
                with self._lock:
                    self._exception.put(e)
        with self._lock:
            self._thread_active[stim_idx] = False

    def lsl_stream_system(
        self,
        stim_idx: int,
        streams: list[DaqStreamDescription],
        daq_func,
        format: int = pylsl.cf_int32,
        require_consumers: bool = True,
        consumer_timeout: float = 10.0,
    ) -> None:
        """Process for starting a Lab Streaming Layer (LSL) to process the data streams from DAQ system.
        One thread reads the (shared) device interface and demultiplexes the packets into one LSL outlet per stream.
        :param stim_idx:        Integer with array index to write into heartbeat feedback array
        :param streams:         List with descriptions of each DAQ stream / LSL outlet
        :param daq_func:        Function to get the next packet from the DAQ device, returning
                                (head_cmd, data, timestamp). For sample streams data is a list[channels] and
                                timestamp a float, for batch streams data is list[samples][channels] and timestamp
                                a list[float]. On timeout / invalid data it returns (None, [], None).
        :param format:          Defined LSL datatype for saving data in the stream
        :param require_consumers:Boolean for requiring at least one consumer on any of the outlets
        :param consumer_timeout:Floating value with timeout [s] for waiting for consumers
        :return:                None
        """
        outlets: dict[int, tuple[StreamOutlet, bool]] = dict()
        for stream in streams:
            outlet = self._establish_lsl_outlet(
                idx=stim_idx,
                lsl_name=stream.name,
                lsl_type="stream_daq",
                sampling_rate=stream.sampling_rate,
                channel_num=len(stream.channel_labels),
                channel_labels=stream.channel_labels,
                channel_units=stream.channel_units,
                channel_layout=stream.channel_layout,
                channel_type=format,
                check_for_consumers=False,
            )
            outlets[stream.head_cmd] = (outlet, stream.is_batch)

        def has_consumers() -> bool:
            return any(outlet.have_consumers() for outlet, _ in outlets.values())

        # Not every stream needs a consumer (e.g. only one is plotted), so wait for any of them
        if require_consumers:
            deadline = monotonic() + consumer_timeout
            while not has_consumers():
                if monotonic() > deadline:
                    raise ConnectionError("no consumers available")
                sleep(0.05)

        while self._event.is_set() and self._is_active:
            try:
                head, data, tb = daq_func()
                if head is None or not data or tb is None:
                    continue
                outlet, is_batch = outlets[head]
                if is_batch:
                    if len(tb) != len(data):
                        continue
                    outlet.push_chunk(x=data, timestamp=tb, pushthrough=True)
                else:
                    outlet.push_sample(x=data, timestamp=tb, pushthrough=True)
                # Heartbeat
                with self._lock:
                    self._thread_active[stim_idx] = (not require_consumers) or has_consumers()
            except Exception as e:
                with self._lock:
                    self._exception.put(e)
        with self._lock:
            self._thread_active[stim_idx] = False

    def lsl_stream_util(self, stim_idx: int, name: str, sampling_rate: float = 2.0) -> None:
        """Process for starting a Lab Streaming Layer (LSL) to process the utilization of the host computer
        :param stim_idx:        Integer with array index to write into heartbeat feedback array
        :param name:            String with name of the LSL stream (must match with a recording process)
        :param sampling_rate:   Float with sampling rate for determining the sampling rate
        :return:                None
        """
        if sampling_rate > 2.0:
            raise ValueError("Please reduce sampling rate lower than 2.0 Hz")

        outlet = self._establish_lsl_outlet(
            idx=stim_idx,
            lsl_name=name,
            lsl_type="utilization",
            sampling_rate=sampling_rate,
            channel_num=2,
            channel_labels=["CPU", "RAM"],
            channel_units=["%", "%"],
            channel_layout=[[1, 2]],
            channel_type=pylsl.cf_float32,
            check_for_consumers=False,
        )

        while self._event.is_set() and self._is_active:
            try:
                cpu = cpu_percent()
                mem = virtual_memory().percent
                outlet.push_sample(x=[cpu, mem], timestamp=0.0, pushthrough=True)
                with self._lock:
                    self._thread_active[stim_idx] = outlet.have_consumers()
                sleep(1 / sampling_rate)
            except Exception as e:
                with self._lock:
                    self._exception.put(e)
        with self._lock:
            self._thread_active[stim_idx] = False

    def lsl_split_stream(self, stim_idx: int, name_in: str, name_out: list[str]) -> None:
        """LSL layer for implementing a 1:N split function
        :param stim_idx:    Integer with array index to write into heartbeat feedback array
        :param name_in:     Name of the LSL StreamInlet for cloning
        :param name_out:    List with LSL StreamOutlet names
        :return:            None
        """
        inlet = self._establish_lsl_inlet(name_in)
        outlets: list[StreamOutlet] = list()
        self._logger.debug(name_out)
        for name in name_out:
            self._logger.debug(f"Include LSL stream splitter: {name_in} -> {name}")
            outlets.append(
                self._establish_lsl_outlet(
                    idx=stim_idx,
                    lsl_name=name,
                    lsl_type=inlet.info().type(),
                    sampling_rate=inlet.info().nominal_srate(),
                    channel_units=inlet.info().get_channel_units(),
                    channel_num=inlet.info().channel_count(),
                    channel_type=inlet.info().channel_format(),
                    channel_labels=inlet.info().get_channel_labels(),
                    channel_layout=inlet.info().get_channel_types(),
                    check_for_consumers=False,
                )
            )
        self._logger.debug(f"Running {len(outlets)} LSL split units")

        while self._event.is_set() and self._is_active:
            try:
                data, time = inlet.pull_sample(timeout=0.5)
                if not data and not time:
                    continue
                for outlet in outlets:
                    outlet.push_sample(x=data, timestamp=time, pushthrough=True)
                with self._lock:
                    self._thread_active[stim_idx] = all([outlet.have_consumers() for outlet in outlets])
            except Exception as e:
                with self._lock:
                    self._exception.put(e)
        with self._lock:
            self._thread_active[stim_idx] = False

    def lsl_process_stream(
        self,
        stim_idx: int,
        name_in: str,
        name_out: str,
        num_samples: int,
        func_init_daq,
        func_process_daq,
        lsl_format: int = 0,
        require_consumers: bool = True,
        channel_labels: list[str] | None = None,
        channel_units: list[str] | None = None,
        channel_layout: list[list[int]] | None = None,
    ) -> None:
        """Function for processing the incoming data from LSL stream
        :param stim_idx:        Integer with array index to write into heartbeat feedback array
        :param name_in:         String with name of the LSL stream to catch it
        :param name_out:        String with name of the LSL stream to push the data to
        :param num_samples:     Number of samples in the chunk for one processing step
        :param func_init_daq:   Function to initialize the DAQ device (needs input: sampling rate)
        :param func_process_daq:Function to process the data from a DAQ device (needs input: list with chunk data, returned: chunk output)
        :param lsl_format:      Integer with lsl datatype format cf_*
        :param require_consumers:Boolean for requiring connected LSL consumers
        :param channel_labels:  Optional list with labels of the output channels (default: labels of the input stream)
        :param channel_units:   Optional list with units of the output channels (default: units of the input stream)
        :param channel_layout:  Optional list with layout of the output channels (default: layout of the input stream)
        :return:                None
        """
        inlet = self._establish_lsl_inlet(name_in)
        info_in = inlet.info()
        sampling_rate = info_in.nominal_srate()
        num_channels = info_in.channel_count()

        if channel_labels is None:
            channel_labels = info_in.get_channel_labels() or [f"CH{idx}" for idx in range(num_channels)]
        if channel_units is None:
            channel_units = info_in.get_channel_units() or [""] * num_channels
        if channel_layout is None:
            channel_layout = info_in.get_channel_types() or [""] * num_channels

        outlet = self._establish_lsl_outlet(
            idx=stim_idx,
            lsl_name=name_out,
            lsl_type=info_in.type(),
            sampling_rate=sampling_rate,
            channel_units=channel_units,
            channel_num=num_channels,
            channel_labels=channel_labels,
            channel_layout=channel_layout,
            channel_type=inlet.info().channel_format() if lsl_format == 0 else lsl_format,
            check_for_consumers=False,
        )
        func_init_daq(sampling_rate)

        while self._event.is_set():
            try:
                data_buf, ts_buf = inlet.pull_chunk(max_samples=num_samples, timeout=0.1)
                if not data_buf or ts_buf is None:
                    continue

                data_out = func_process_daq(data_buf)
                outlet.push_chunk(x=data_out, timestamp=ts_buf, pushthrough=True)
                with self._lock:
                    self._thread_active[stim_idx] = (not require_consumers) or outlet.have_consumers()
            except Exception as e:
                with self._lock:
                    self._exception.put(e)
        with self._lock:
            self._thread_active[stim_idx] = False

    def lsl_stream_check_equality(self, stim_idx: int, names: list[str], num_samples: int = 16) -> None:
        """LSL Layer for checking content of different LSL StreamInlets
        :param stim_idx:    Integer with array index to write into heartbeat feedback array
        :param names:       List with LSL stream names to check the data
        :param num_samples: Integer for pulling N samples in one chunk
        :return:            None
        """
        inlets: list[StreamInlet] = list()
        for name in names:
            inlets.append(self._establish_lsl_inlet(name))

        while self._event.is_set() and self._is_active:
            data = list()
            time = list()
            for inlet in inlets:
                data_new, time_new = inlet.pull_chunk(max_samples=num_samples, timeout=0.1)
                if not data_new or not time_new:
                    break

                data.append(data_new)
                time.append(time_new)
            if not data or not time:
                continue

            with self._lock:
                self._thread_active[stim_idx] = True

            self._logger.debug(f"{time}: {data}")
            for val, ts in zip(data[1:], time[1:]):
                assert data[0] == val
                assert time[0] == ts

    def lsl_record_stream(
        self,
        stim_idx: int,
        names: list[str],
        path2save: Path | str,
        extra_attrs: dict[str, str | int | float] | None = None,
    ) -> None:
        """Function for recording and saving the data pushed on LSL stream
        :param stim_idx:            Integer with array index to write into heartbeat feedback array
        :param names:               List with names of the LSL inlet for catching data (0: sensor stream, 1...N: additional infos)
        :param path2save:           Path to save the data (if it is a string, it will be auto-converted)
        :param extra_attrs:         Optional dictionary with additional attributes to write into the file (e.g. processing info)
        :return: None
        """
        if type(names) is not list:
            raise TypeError("names must be a list")

        inlets: list[StreamInlet] = list()
        for name in names:
            inlets.append(self._establish_lsl_inlet(name))
        # Extract meta
        data_format = inlets[0].info().channel_format()
        time = datetime.today().strftime("%Y%m%d_%H%M%S")

        stream_time = list()
        stream_data = list()

        path2file = Path(path2save) / f"{time}_{names[0]}.h5"
        path2file.parent.mkdir(parents=True, exist_ok=True)
        with File(path2file.as_posix(), "w") as f:
            f.attrs["creation_date"] = datetime.today().strftime("%Y-%m-%d")
            f.attrs["data_format"] = data_format
            for key, value in (extra_attrs or {}).items():
                f.attrs[key] = value

            for idx, (name, inlet) in enumerate(zip(names, inlets)):
                stream_time.append(f.create_dataset(f"{name}_time", (0,), maxshape=(None,), dtype=float))
                stream_time[idx].attrs["unit"] = "s"

                channel = inlet.info().channel_count()
                format = self._get_h5_format(inlet.info().channel_format())
                stream_data.append(
                    f.create_dataset(f"{name}_data", (0, channel), maxshape=(None, channel), dtype=format)
                )
                stream_data[idx].attrs["unit"] = self._to_str_list(inlet.info().get_channel_units(), channel)
                stream_data[idx].attrs["label"] = self._to_str_list(inlet.info().get_channel_labels(), channel)
                stream_data[idx].attrs["layout"] = self._to_str_list(inlet.info().get_channel_types(), channel)
                stream_data[idx].attrs["sampling_rate"] = inlet.info().nominal_srate()
                stream_data[idx].attrs["channel_count"] = inlet.info().channel_count()
                stream_data[idx].attrs["type"] = inlet.info().type()
            f.flush()

            while self._event.is_set() and self._is_active:
                try:
                    for inlet, time, data in zip(inlets, stream_time, stream_data):
                        data_new, time_new = inlet.pull_chunk(
                            max_samples=self._get_number_stream_samples(inlet.info().nominal_srate()),
                            timeout=0.01,
                        )
                        if not data_new and not time_new:
                            continue
                        else:
                            with self._lock:
                                self._thread_active[stim_idx] = True
                            idx = len(time)
                            new = len(time_new)
                            time.resize((idx + new,))
                            time[idx : idx + new] = time_new
                            data.resize((idx + new, inlet.info().channel_count()))
                            data[idx : idx + new, :] = np.asarray(data_new)[:, :]
                            f.flush()
                except Exception as e:
                    self._exception.put(e)
            f.close()
            with self._lock:
                self._thread_active[stim_idx] = False

    def lsl_plot_stream(
            self,
            plot_config: LivePlotterChannelConfig,
            window_length: float = 10.0,
    ) -> None:
        """Function for LSL to enable live plotting using LivePlotter (PyQtGraph)
        receiving a pre-built LivePlotterChannelConfig object.
        """
        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication(sys.argv)

        self._plotter = LivePlotter(plot_config, window_width_sec=window_length)
        try:
            app.exec()
        finally:
            self._plotter = None
