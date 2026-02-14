import numpy as np
from logging import getLogger, Logger
from h5py import File
from datetime import datetime
from pathlib import Path
from time import sleep
from tqdm import tqdm
from threading import Event, Thread, Lock
from psutil import cpu_percent, virtual_memory
from pylsl import (
    StreamInfo,
    StreamInlet,
    StreamOutlet,
    resolve_bypred,
    cf_int16,
    cf_int32,
    cf_float32,
    proc_threadsafe
)
from queue import Queue, Empty
from vispy import app, scene, visuals
from api.data_api import DataAPI, RawRecording


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
            self._data0[self._index:, 0] = tb
            self._data0[self._index:, 1] = value
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
    _num_missed: int=0

    def __init__(self) -> None:
        """Class for managing all threads for the LSL data processing
        :return:    None
        """
        self._logger = getLogger(__name__)
        self._event = Event()
        self._lock = Lock()
        self._exception = Queue()
        self._release_threads()
        self._channel_plots = []
        self.x_counter = 0

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

    def register(self, func, args) -> None:
        """Registering a thread with custom instruction
        :param func:    Function object for further processing in own thread
        :param args:    Arguments of the object for starting it
        :return:        None
        """
        if not len(self._thread):
            self._thread = [Thread(target=self._thread_watchdog_heartbeat, args=())]
        self._thread.append(Thread(target=func, args=args))

    def start(self) -> None:
        """
        Starting all threads including heartbeat watchdog in own thread
        :return:    None
        """
        self._thread_active = list()
        if len(self._thread) < 2:
            raise AssertionError("No threads registered")
        else:
            self._thread_active = [False for _ in self._thread[1:]]
            self._num_missed = 0
            self._is_active = True
            self._event.set()
            for idx, p in enumerate(self._thread):
                p.start()
            sleep(0.2)

    def stop(self) -> None:
        """Stopping all threads and waiting for shutdown all threads
        :return:        None
        """
        self._event.clear()
        for p in self._thread:
            p.join(timeout=1.)
        self._release_threads()

    def check_exception(self) -> None:
        """Function for checking if any exception information is available from any thread, and return it
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
        sleep(1.)
        for _ in tqdm(range(int(wait_time_sec))):
            if self._is_active:
                self.check_exception()
                sleep(1.)
            else:
                raise RuntimeError(f"One thread is shutdown [{self._is_active}] - {self._thread_active}")

    def _establish_lsl_outlet(self, idx: int, lsl_name: str, lsl_type: str,  sampling_rate: float, channel_num: int, channel_type: int=cf_int16) -> tuple[StreamOutlet, StreamInfo]:
        info = StreamInfo(
            name=lsl_name,
            type=lsl_type,
            channel_count=channel_num,
            nominal_srate=sampling_rate,
            channel_format=channel_type,
            source_id=f"{lsl_name}_uid"
        )
        outlet = StreamOutlet(info)
        while not outlet.wait_for_consumers(timeout=30.0):
            with self._lock:
                self._thread_active[idx] = True
            sleep(0.25)
        return outlet, info

    @staticmethod
    def _establish_lsl_inlet(name: str) -> StreamInlet:
        info = resolve_bypred(
            predicate=f"name='{name}'",
            minimum=1,
            timeout=1.
        )
        if not info:
            raise ValueError(f"LSL stream with {name} not found")
        inlet = StreamInlet(
            info=info[0],
            max_buflen=60,
            max_chunklen=1024,
            recover=True,
            processing_flags=proc_threadsafe
        )
        return inlet

    def _release_threads(self) -> None:
        self._logger.debug(f"Empty thread")
        self._thread = []
        self._thread_active = []
        self._is_active = False

    def _thread_watchdog_heartbeat(self) -> None:
        while self._event.is_set():
            check_alive = all([thread.is_alive() for thread in self._thread[1:]])
            try:
                with self._lock:
                    checker = all(self._thread_active)
                    for i, _ in enumerate(self._thread_active):
                        self._thread_active[i] = False
                if check_alive and checker:
                    self._num_missed = 0
                else:
                    self._num_missed += 1
                if self._num_missed >= 5:
                    self._is_active = False
            except Exception as e:
                with self._lock:
                    self._exception.put(e)
            sleep(2.)

    def _thread_dummy(self, stime_idx: int) -> None:
        while self._event.is_set():
            try:
                with self._lock:
                    self._thread_active[stime_idx] = True
                sleep(0.1)
            except Exception as e:
                with self._lock:
                    self._exception.put(e)

    def lsl_stream_mock(self, stim_idx: int, name: str, channel_num: int=2, sampling_rate: float=200.) -> None:
        """Process for starting a Lab Streaming Layer (LSL) to mock the DAQ hardware with random data
        :param stim_idx:        Integer with array index to write into heartbeat feedback array
        :param name:            String with name of the LSL stream (must match with recording process)
        :param channel_num:     Channel number to start stream from
        :param sampling_rate:   Floating value with sampling rate in Hz
        :return:                None
        """
        outlet = self._establish_lsl_outlet(
            idx=stim_idx,
            lsl_name=name,
            lsl_type='mock_sinusoidal_daq',
            sampling_rate=sampling_rate,
            channel_num=channel_num,
            channel_type=cf_int16
        )[0]

        ite_num = 0
        chck_num = np.random.randint(low=10, high=int(0.5 * sampling_rate))
        while self._event.is_set():
            try:
                # Heartbeat
                if ite_num >= chck_num:
                    ite_num = 0
                    with self._lock:
                        self._thread_active[stim_idx] = outlet.have_consumers()
                else:
                    ite_num += 1
                # Process data
                outlet.push_sample(
                    x=np.random.randint(low=-2**15, high=2**15, size=channel_num).tolist(),
                    timestamp=0.0,
                    pushthrough=True
                )
                sleep(1 / sampling_rate)
            except Exception as e:
                with self._lock:
                    self._exception.put(e)

    def lsl_stream_file(self, stim_idx: int, name: str, path2data: str, file_index: int=0, prefix: str= 'data') -> None:
        """Process for starting a Lab Streaming Layer (LSL) to stream the file content into the DAQ system by mocking the DAQ hardware
        :param stim_idx:        Integer with array index to write into heartbeat feedback array
        :param name:            String with name of the LSL stream (must match with recording process)
        :param path2data:       Path to folder with pre-recorded data files
        :param file_index:      Number of the file in the folder to process
        :param prefix:          Prefix of the data file to find
        :return:                None
        """
        path0 = Path(path2data)
        if not path0.exists():
            raise AttributeError("File is not available")
        data: RawRecording = DataAPI(path2data=path0, data_prefix=prefix).read_data_file(file_index)

        outlet = self._establish_lsl_outlet(
            idx=stim_idx,
            lsl_name=name,
            lsl_type='mock_file_daq',
            sampling_rate=float(data.sampling_rate),
            channel_num=int(data.num_channels),
            channel_type=cf_int16
        )[0]

        while self._event.is_set():
            try:
                for stime, sdata in zip(data.time, data.data.T):
                    if not self._event.is_set():
                        break
                    # Heartbeat
                    with self._lock:
                        self._thread_active[stim_idx] = outlet.have_consumers()
                    # Process data
                    outlet.push_sample(
                        x=sdata.tolist(),
                        timestamp=stime,
                        pushthrough=True
                    )
                    sleep(1 / data.sampling_rate)
            except Exception as e:
                with self._lock:
                    self._exception.put(e)

    def lsl_stream_data(self, stim_idx: int, name: str, daq_func, channel_num: int, sampling_rate: float) -> None:
        """Process for starting a Lab Streaming Layer (LSL) to process the data stream from DAQ system
        :param stim_idx:        Integer with array index to write into heartbeat feedback array
        :param name:            String with name of the LSL stream (must match with recording process)
        :param daq_func:        Function to get data from DAQ device (returned list)
        :param channel_num:     Channel number to start stream from
        :param sampling_rate:   Floating value with sampling rate in Hz
        :return:                None
        """
        outlet = self._establish_lsl_outlet(
            idx=stim_idx,
            lsl_name=name,
            lsl_type='sensor_data',
            sampling_rate=sampling_rate,
            channel_num=channel_num,
            channel_type=cf_int32
        )[0]

        use_batch_mode = 'batch' in daq_func.__name__
        if use_batch_mode:
            while self._event.is_set():
                try:
                    # Data Processing
                    data, tb = daq_func()
                    if len(data) != len(tb):
                        continue
                    # Heartbeat
                    with self._lock:
                        self._thread_active[stim_idx] = outlet.have_consumers()
                    outlet.push_chunk(
                        x=data,
                        timestamp=tb,
                        pushthrough=True
                    )
                except Exception as e:
                    with self._lock:
                        self._exception.put(e)
        else:
            while self._event.is_set():
                try:
                    # Data Processing
                    data, tb = daq_func()
                    if not data or tb is None:
                        continue
                    # Heartbeat
                    with self._lock:
                        self._thread_active[stim_idx] = outlet.have_consumers()
                    outlet.push_sample(
                        x=data,
                        timestamp=tb,
                        pushthrough=True
                    )
                except Exception as e:
                    with self._lock:
                        self._exception.put(e)

    def lsl_stream_util(self, stim_idx: int, name: str, sampling_rate: float=2.) -> None:
        """Process for starting a Lab Streaming Layer (LSL) to process the utilization of the host computer
        :param stim_idx:        Integer with array index to write into heartbeat feedback array
        :param name:            String with name of the LSL stream (must match with recording process)
        :param sampling_rate:   Float with sampling rate for determining the sampling rate
        :return:                None
        """
        if sampling_rate > 2.:
            raise ValueError("Please reduce sampling rate lower than 2.0 Hz")

        outlet = self._establish_lsl_outlet(
            idx=stim_idx,
            lsl_name=name,
            lsl_type='utilization',
            sampling_rate=sampling_rate,
            channel_num=2,
            channel_type=cf_float32
        )[0]

        while self._event.is_set():
            try:
                with self._lock:
                    self._thread_active[stim_idx] = outlet.have_consumers()
                outlet.push_sample(
                    x=[cpu_percent(), virtual_memory().percent],
                    timestamp=0.0,
                    pushthrough=True
                )
                sleep(1 / sampling_rate)
            except Exception as e:
                with self._lock:
                    self._exception.put(e)

    def lsl_record_stream(self, stim_idx: int, name: str, path2save: Path | str) -> None:
        """Function for recording and saving the data pushed on LSL stream
        :param stim_idx:            Integer with array index to write into heartbeat feedback array
        :param name:                String with name of the LSL stream in order to catch it
        :param path2save:           Path to save the data (if it is a string, it will be auto-converted)
        :return: None
        """
        path = Path(path2save) if type(path2save) == str else path2save
        inlet = self._establish_lsl_inlet(name)
        # Extract meta
        channels = inlet.info().channel_count()
        sampling_rate = inlet.info().nominal_srate()
        sys_type = inlet.info().type()
        data_format = inlet.info().channel_format()
        time = datetime.today().strftime('%Y%m%d_%H%M%S')

        if not path.is_dir():
            path.mkdir(parents=True, exist_ok=True)
        with File(path.absolute() / f"{time}_{name}.h5", "w") as f:
            f.attrs["sampling_rate"] = sampling_rate
            f.attrs["channel_count"] = channels
            f.attrs["type"] = sys_type
            f.attrs["creation_date"] = datetime.today().strftime('%Y-%m-%d')
            f.attrs["data_format"] = data_format
            ts_dset = f.create_dataset("time", (0,), maxshape=(None,), dtype=float)
            ts_dset.attrs["unit"] = "s"
            match data_format:
                case 1: #cf_float32
                    format_h5 = "float32"
                case 2: #cf_double64
                    format_h5 = "float64"
                case 3: #cf_string
                    format_h5 = "string"
                case 4:  # cf_int32
                    format_h5 = "int32"
                case 5:  # cf_int16
                    format_h5 = "uint16"
                case 6:  # cf_int8
                    format_h5 = "uint8"
                case 7:  # cf_int64
                    format_h5 = "int64"
                case _:
                    raise ValueError(f"Unknown LSL datatype format")
            data_dset = f.create_dataset("data", (0, channels), maxshape=(None, channels), dtype=format_h5)
            ts_dset.attrs["unit"] = ""
            f.flush()

            process_list = [i for i in range(channels)]
            cnt_flush = 0
            max_samples = int(sampling_rate / 50) if sampling_rate > 500. else 10
            while self._event.is_set():
                try:
                    data_buf, ts_buf = inlet.pull_chunk(
                        max_samples=max_samples,
                        timeout=10e-3
                    )
                    if not data_buf:
                        continue
                    else:
                        with self._lock:
                            self._thread_active[stim_idx] = True
                        idx = len(ts_dset)
                        new = len(ts_buf)
                        ts_dset.resize((idx + new,))
                        data_dset.resize((idx + new, channels))
                        ts_dset[idx:idx + new] = ts_buf
                        data_dset[idx:idx + new, :] = np.asarray(data_buf)[:, process_list]
                        if cnt_flush == 3:
                            f.flush()
                            cnt_flush = 0
                        else:
                            cnt_flush += 1
                except Exception as e:
                    self._exception.put(e)
            f.close()

    def lsl_plot_stream(
            self, stim_idx: int, name: str, window_length: float = 10., update_rate: float = 12.
    ) -> None:
        """Function for LSL to enable live plotting of the incoming results using VisPy
        :param stim_idx:        Integer with array index to write into heartbeat feedback array
        :param name:            String with name of the LSL stream to get data
        :param window_length:   Floating value with length of time window for plotting in seconds
        :param update_rate:     Floating value with update rate of the LSL datastream
        :return:                None
        """
        line_color = ['red', 'green', 'blue', 'lime']
        mode_util = 'util' in name
        inlet = self._establish_lsl_inlet(name)
        # --- Extract meta
        channels = inlet.info().channel_count()
        sampling_rate = inlet.info().nominal_srate()
        if sampling_rate > 4500.:
            raise AttributeError(f"Sampling rate {sampling_rate} is too high")
        # --- Build ring buffer and update func
        max_samples = int(sampling_rate / 50) if sampling_rate > 500. else 10
        number_samples_window = int(window_length * sampling_rate)
        buffer_lsl = [RingBuffer(number_samples_window) for _ in range(channels)]
        buffer_gpu = buffer_lsl.copy()
        # --- Build app
        canvas = scene.SceneCanvas(
            size=(800, 150 + channels*120),
            title=f"Live Plot @{sampling_rate} Hz ({name})",
            keys=None,
            app="glfw",
            show=True,
        )

        grid = canvas.central_widget.add_grid(spacing=5)
        x_range = (0, number_samples_window-1)
        y_range = (0, 65535) if not mode_util else (0, 100)

        views = []
        lines = []
        status_texts = []
        fps_texts = []

        for ch in range(channels):
            view = grid.add_view(row=ch, col=1, camera='panzoom')
            view.camera.set_range(x=x_range, y=y_range)
            views.append(view)

            yaxis = scene.AxisWidget(orientation='left')
            xaxis = scene.AxisWidget(orientation='bottom')

            grid.add_widget(yaxis, row=ch, col=0)
            grid.add_widget(xaxis, row=ch, col=1)

            yaxis.link_view(view)
            #xaxis.link_view(view)

            data = buffer_gpu[ch].get_data()
            line = scene.visuals.Line(pos=data, color=line_color[ch % len(line_color)],
                                      width=2, parent=view.scene)
            lines.append(line)

            scene.visuals.Text(
                text=f"Ch {ch+1}",
                parent=view.scene,
                pos=(20, 0.5),
                anchor_x='right',
                anchor_y='center',
                color='white',
                font_size=10,
            )

            status_text = scene.visuals.Text(
               text="LSL: OK",
               parent=view.scene,
               color='green',
               pos=(0.95 * number_samples_window, -20),
               font_size=8
            )
            status_texts.append(status_text)

            fps_text = scene.visuals.Text(
               text="FPS: 0",
               color='green',
               parent=view.scene,
               pos=(0.85 * number_samples_window, -20),
               font_size=8
            )
            fps_texts.append(fps_text)

        def update_plot_data():
            nonlocal buffer_lsl
            while self._event.is_set():
                try:
                    samples, _ = inlet.pull_chunk(
                        max_samples=max_samples,
                        timeout=10e-3
                    )
                    if not samples:
                        continue
                    else:
                        with self._lock:
                            self._thread_active[stim_idx+1] = True
                        for sample in samples:
                            for ch, value in enumerate(sample):
                                buffer_lsl[ch].append(value)
                except Exception as e:
                    with self._lock:
                        self._exception.put(e)

        def update_plot_canvas(events):
            nonlocal buffer_gpu
            if not self._event.is_set():
                app.quit()

            buffer_gpu = buffer_lsl

            y_min = np.inf
            y_max = -np.inf

            #self.x_counter += 1

            for ch in range(channels):
                data = buffer_gpu[ch].get_data()
                lines[ch].set_data(data)

                y = data[:, 1]
                y_min = min(y_min, y.min())
                y_max = max(y_max, y.max())

                if y_max > y_min:
                   pad = (y_max - y_min) * 0.1
                else:
                    pad = 1.0

                views[ch].camera.set_range(
                   x=(0, number_samples_window-1),
                   y=(y_min - pad, y_max + 1))

            for ch, status_text in enumerate(status_texts):
                status_text.pos = (0.85 * number_samples_window, -20)
                if not self._is_active:
                    status_text.text = "LSL: DEAD"
                    status_text.color = 'red'

            for ch, fps_text in enumerate(fps_texts):
                fps_text.pos = (0.95 * number_samples_window, -20)


        def update_on_fps(fps):
            with self._lock:
                self._thread_active[stim_idx] = fps > 0
            fps_text.text = f"FPS: {fps:.1f}"

        # --- Starting the process
        self.register(func=update_plot_data, args=())
        self._thread_active.extend([0])
        self._thread[-1].start()
        canvas.measure_fps(callback=update_on_fps)
        app.Timer(
            interval=1/update_rate,
            connect=update_plot_canvas,
            start=True
        )
        app.run()
