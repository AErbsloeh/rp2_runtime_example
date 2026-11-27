from collections import deque
from collections.abc import Callable
from logging import getLogger
from threading import Event, Thread
from time import monotonic, sleep

import numpy as np
from pylsl import StreamInfo, StreamInlet, StreamOutlet, cf_int32, local_clock, resolve_byprop


def _create_outlet(
    name: str, sampling_rate: float, num_channels: int, label: str, unit: str, stream_type: str
):
    info = StreamInfo(
        name=name,
        type=stream_type,
        channel_count=num_channels,
        nominal_srate=sampling_rate,
        channel_format=cf_int32,
        source_id=f"{name}_uid",
    )
    info.set_channel_labels(
        [f"{label}{idx}" if num_channels > 1 else label for idx in range(num_channels)]
    )
    info.set_channel_units([unit] * num_channels)
    info.set_channel_types([stream_type] * num_channels)
    return StreamOutlet(info)


class LslGenerator:
    """Generating test data in a LSL stream (int32), e.g. as source for the DigitalTwin

    Args:
        name (str): Name of the LSL stream
        sampling_rate (float): Sampling rate in Hz
        func (Callable): Function with the time in seconds (np.ndarray) as input and the values as output, 1D for one channel or [samples, channels]
        label (str, optional): Label of the channel. Defaults to "data"
        unit (str, optional): Unit of the channel. Defaults to "LSB"
        realtime (bool, optional): True to push the samples with the clock of the nominal sampling rate. False to push a block only if the consumer requests it with `push_block`
            (e.g. a `DeviceLoopback` with this generator), as fast as the consumer works and without lost samples; the timestamps are the time of the signal (start time + index / sampling rate). Defaults to True
    """

    def __init__(
        self,
        name: str,
        sampling_rate: float,
        func: Callable[[np.ndarray], np.ndarray],
        label: str = "data",
        unit: str = "LSB",
        realtime: bool = True,
    ) -> None:
        self._fs = sampling_rate
        self._func = func
        self._realtime = realtime
        self._stop = Event()
        self._thread: Thread | None = None
        self._num_pushed = 0
        self._time_start: float | None = None
        probe = np.asarray(func(np.zeros(1)))
        num_channels = 1 if probe.ndim == 1 else probe.shape[1]
        self._outlet = _create_outlet(name, sampling_rate, num_channels, label, unit, "data")

    @property
    def realtime(self) -> bool:
        """True if the generator pushes the samples with the clock, False if the consumer requests the blocks"""
        return self._realtime

    def start(self) -> None:
        """Start the clock of the generator, nothing to do if it is not real time (the consumer requests the blocks)"""
        if self._thread is not None or not self._realtime:
            return
        self._stop.clear()
        self._thread = Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    def push_block(self, num_samples: int) -> None:
        """Push the next block of the signal with the time of the signal as timestamps (only if not real time)
        :param num_samples: Number of samples of the block
        :return:            None
        """
        if self._realtime:
            raise RuntimeError("push_block needs a generator with realtime=False")
        if self._time_start is None:
            self._time_start = local_clock()
        indices = self._num_pushed + np.arange(num_samples)
        values = np.asarray(self._func(indices / self._fs)).astype(np.int32)
        self._outlet.push_chunk(
            values.reshape(num_samples, -1).tolist(), (self._time_start + indices / self._fs).tolist()
        )
        self._num_pushed += num_samples

    def _run(self) -> None:
        t_start = monotonic()
        num_sent = 0
        while not self._stop.is_set():
            num_due = int((monotonic() - t_start) * self._fs) - num_sent
            if num_due > 0:
                values = np.asarray(self._func((num_sent + np.arange(num_due)) / self._fs)).astype(
                    np.int32
                )
                self._outlet.push_chunk(values.reshape(num_due, -1).tolist())
                num_sent += num_due
            sleep(0.005)


class DeviceLoopback:
    """Sending the samples of a LSL stream one by one to a device and publishing the returned values as new LSL stream
    with the timestamps of the source, so the original and the returned data can be compared (e.g. in the plot of the DigitalTwin). Only for streams with one channel

    Args:
        process (Callable): Function with one integer sample as input and the result of the device as output or None for no response. The sample is skipped on None
        read_results (Callable, optional): Function which returns the results of the device as [samples][channels] (e.g. `DeviceAPI().read_processing_results`), if the device sends them delayed and in blocks
            and not as return value of `process`. The results are assigned to the source samples in the order of sending, so the device must return exactly one result per sample.
            It is called once before the first sample is sent, so the device is prepared for the results (e.g. the streams are registered) and old results are discarded. Defaults to None
        name_in (str): Name of the LSL stream with the data to send
        name_out (str, optional): Name of the LSL stream with the returned data. Defaults to "DeviceLoopback"
        label (str, optional): Label of the returned channel. Defaults to "device"
        unit (str, optional): Unit of the returned channel. Defaults to "LSB"
        realtime (bool, optional): True to send the samples as they arrive in the stream (the device must keep up with the source). False to run without real time: the loopback requests a block
            from the generator, sends it to the device, waits for all results and only then requests the next block, so the stream is continuous and nothing is lost. Defaults to True
        generator (LslGenerator, optional): Generator of the stream `name_in` with realtime=False, needed if not real time. Defaults to None
        block_samples (int, optional): Number of samples of one block if not real time, with `read_results` a multiple of the block size of the output stream of the device (`num_samples` of `daq_out` in the firmware). Defaults to 64
        timeout_sec (float, optional): Time in seconds without new samples or results until the loopback stops with an error if not real time. Defaults to 2.0
    """

    def __init__(
        self,
        process: Callable[[int], int | None],
        name_in: str,
        name_out: str = "DeviceLoopback",
        label: str = "device",
        unit: str = "LSB",
        read_results: Callable[[], list[list[int]]] | None = None,
        realtime: bool = True,
        generator: LslGenerator | None = None,
        block_samples: int = 64,
        timeout_sec: float = 2.0,
    ) -> None:
        if not realtime and (generator is None or generator.realtime):
            raise ValueError("Without real time the loopback needs a generator with realtime=False")
        self._process = process
        self._read_results = read_results
        self._realtime = realtime
        self._generator = generator
        self._block_samples = block_samples
        self._timeout_sec = timeout_sec
        self._pending_timestamps: deque[float] = deque()
        self._num_unassigned = 0
        self._stop = Event()
        self._thread: Thread | None = None
        self._exception: Exception | None = None
        self._logger = getLogger(__name__)
        self._round_trip_sec: list[float] = []
        self._num_no_response = 0
        streams = resolve_byprop("name", name_in, timeout=10.0)
        if not streams:
            raise ValueError(f"LSL stream with {name_in} not found")
        self._inlet = StreamInlet(streams[0], max_buflen=60, recover=True)
        info_in = self._inlet.info()
        if info_in.channel_count() != 1:
            raise ValueError("Only streams with one channel can be sent to the device")
        self._outlet = _create_outlet(name_out, info_in.nominal_srate(), 1, label, unit, info_in.type())

    @property
    def exception(self) -> Exception | None:
        """Error which stopped the loopback (e.g. the device does not respond), None while it works"""
        return self._exception

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._exception = None
        self._thread = Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the loopback and log the mean round trip time (limits the sampling rate of the source)"""
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        if self._round_trip_sec:
            mean_sec = float(np.mean(self._round_trip_sec))
            self._logger.info(
                f"Device loopback: {len(self._round_trip_sec)} samples, mean round trip {1e3 * mean_sec:.2f} ms "
                f"(max. sampling rate about {1.0 / mean_sec:.0f} Hz), {self._num_no_response} without response, "
                f"{self._num_unassigned} results without source sample, {len(self._pending_timestamps)} samples without result"
            )

    def _run(self) -> None:
        try:
            if self._read_results is not None:
                self._read_results()
            if self._realtime:
                self._run_realtime()
            else:
                self._run_synchronous()
        except Exception as exception:
            self._exception = exception
            self._logger.error(f"Device loopback stopped: {exception}")

    def _assign_results(self, rows: list[list[int]], returned: list, returned_timestamps: list) -> None:
        for row in rows:
            if not self._pending_timestamps:
                self._num_unassigned += 1
                continue
            returned.append([int(row[0])])
            returned_timestamps.append(self._pending_timestamps.popleft())

    def _send(self, sample: list, timestamp: float, returned: list, returned_timestamps: list) -> None:
        t_start = monotonic()
        result = self._process(int(sample[0]))
        self._round_trip_sec.append(monotonic() - t_start)
        if self._read_results is not None:
            self._pending_timestamps.append(timestamp)
        elif result is None:
            self._num_no_response += 1
        else:
            returned.append([result])
            returned_timestamps.append(timestamp)

    def _run_realtime(self) -> None:
        while not self._stop.is_set():
            data, timestamps = self._inlet.pull_chunk(
                max_samples=16, timeout=0.02 if self._read_results is not None else 0.1
            )
            if not data and self._read_results is None:
                continue
            returned: list = []
            returned_timestamps: list = []
            for sample, timestamp in zip(data, timestamps):
                self._send(sample, timestamp, returned, returned_timestamps)
            if self._read_results is not None:
                self._assign_results(self._read_results(), returned, returned_timestamps)
            if returned:
                self._outlet.push_chunk(returned, returned_timestamps)

    def _run_synchronous(self) -> None:
        self._inlet.open_stream(timeout=self._timeout_sec)
        while not self._stop.is_set():
            self._generator.push_block(self._block_samples)
            block = self._pull_block()
            if block is None:
                return
            returned: list = []
            returned_timestamps: list = []
            for sample, timestamp in zip(*block):
                self._send(sample, timestamp, returned, returned_timestamps)
            self._wait_for_results(returned, returned_timestamps)
            if returned:
                self._outlet.push_chunk(returned, returned_timestamps)

    def _pull_block(self) -> tuple[list, list] | None:
        data: list = []
        timestamps: list = []
        deadline = monotonic() + self._timeout_sec
        while len(data) < self._block_samples:
            if self._stop.is_set():
                return None
            chunk, stamps = self._inlet.pull_chunk(
                max_samples=self._block_samples - len(data), timeout=0.1
            )
            if chunk:
                data.extend(chunk)
                timestamps.extend(stamps)
                deadline = monotonic() + self._timeout_sec
            elif monotonic() > deadline:
                raise TimeoutError(
                    f"The generator delivered {len(data)} of {self._block_samples} samples of the block"
                )
        return data, timestamps

    def _wait_for_results(self, returned: list, returned_timestamps: list) -> None:
        if self._read_results is None:
            return
        deadline = monotonic() + self._timeout_sec
        while self._pending_timestamps and not self._stop.is_set():
            rows = self._read_results()
            if rows:
                self._assign_results(rows, returned, returned_timestamps)
                deadline = monotonic() + self._timeout_sec
            elif monotonic() > deadline:
                raise TimeoutError(
                    f"The device returned no results for {len(self._pending_timestamps)} samples, "
                    "check if the block size is a multiple of the block size of the output stream"
                )


class ErrorMonitor:
    """Publishing the running error between the device and the twin (device minus twin) as new LSL stream (int32)
    The samples of both streams are assigned by their timestamps (the twin and the `DeviceLoopback` use the timestamps of the source), so it works with the delayed results of the device. A sample without partner in the other stream is skipped

    Args:
        name_device (str): Name of the LSL stream with the data returned from the device (e.g. the `name_out` of the `DeviceLoopback`)
        name_twin (str): Name of the LSL stream with the data of the twin
        name_out (str, optional): Name of the LSL stream with the error. Defaults to "TwinError"
        label (str, optional): Label of the channels of the error stream. Defaults to "error"
        unit (str, optional): Unit of the error stream. Defaults to "LSB"
        max_pending (int, optional): Maximum number of samples waiting for the partner of the other stream. Defaults to 100000
        realtime (bool, optional): True if the streams run with the clock of the nominal sampling rate. False if they run faster (without real time), then the monitor reads larger chunks to keep up with the streams. Defaults to True
    """

    def __init__(
        self,
        name_device: str,
        name_twin: str,
        name_out: str = "TwinError",
        label: str = "error",
        unit: str = "LSB",
        max_pending: int = 100000,
        realtime: bool = True,
    ) -> None:
        self._pull_size = 64 if realtime else 4096
        self._name_device = name_device
        self._name_twin = name_twin
        self._name_out = name_out
        self._label = label
        self._unit = unit
        self._max_pending = max_pending
        self._stop = Event()
        self._thread: Thread | None = None
        self._logger = getLogger(__name__)
        self._num_samples = 0
        self._num_skipped = 0
        self._max_abs_error = 0
        self._sum_abs_error = 0

    @property
    def max_abs_error(self) -> int:
        """Largest absolute error since the start"""
        return self._max_abs_error

    @property
    def mae(self) -> float:
        """Running mean absolute error of all compared samples since the start (0 without samples)"""
        num_samples = self._num_samples
        return self._sum_abs_error / num_samples if num_samples else 0.0

    def start(self) -> None:
        """Connect to both LSL streams (they must exist, e.g. start the twin first) and publish the error stream"""
        if self._thread is not None:
            return
        inlets = []
        for name in (self._name_device, self._name_twin):
            streams = resolve_byprop("name", name, timeout=10.0)
            if not streams:
                raise ValueError(f"LSL stream with {name} not found")
            inlets.append(StreamInlet(streams[0], max_buflen=60, recover=True))
        self._inlet_device, self._inlet_twin = inlets
        for inlet in inlets:
            inlet.open_stream(timeout=10.0)
        info_twin = self._inlet_twin.info()
        self._num_channels = info_twin.channel_count()
        if self._inlet_device.info().channel_count() != self._num_channels:
            raise ValueError("The streams of device and twin need the same number of channels")
        self._sampling_rate = info_twin.nominal_srate()
        self._outlet = _create_outlet(
            self._name_out,
            self._sampling_rate,
            self._num_channels,
            self._label,
            self._unit,
            info_twin.type(),
        )
        self._stop.clear()
        self._thread = Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the monitor and log the number of compared samples and the largest error"""
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
            self._logger.info(
                f"Error monitor: {self._num_samples} samples compared, MAE {self.mae:.3f}, max. abs. error {self._max_abs_error}, "
                f"{self._num_skipped} samples without partner"
            )

    @staticmethod
    def match(
        pending_device: deque, pending_twin: deque, tolerance_sec: float
    ) -> tuple[list[list[int]], list[float], int]:
        """Assigning the waiting samples of the device and the twin by their timestamps and removing them from the queues
        :param pending_device:  Queue with tuples (timestamp, sample) of the device
        :param pending_twin:    Queue with tuples (timestamp, sample) of the twin
        :param tolerance_sec:   Largest time difference of two samples which belong together
        :return:                Tuple with the errors (device minus twin) as [samples][channels], their timestamps and the number of skipped samples
        """
        errors: list[list[int]] = []
        timestamps: list[float] = []
        num_skipped = 0
        while pending_device and pending_twin:
            time_device, sample_device = pending_device[0]
            time_twin, sample_twin = pending_twin[0]
            if abs(time_device - time_twin) <= tolerance_sec:
                errors.append([int(round(dev - twin)) for dev, twin in zip(sample_device, sample_twin)])
                timestamps.append(time_twin)
                pending_device.popleft()
                pending_twin.popleft()
            elif time_device < time_twin:
                pending_device.popleft()
                num_skipped += 1
            else:
                pending_twin.popleft()
                num_skipped += 1
        return errors, timestamps, num_skipped

    def _run(self) -> None:
        pending_device: deque = deque(maxlen=self._max_pending)
        pending_twin: deque = deque(maxlen=self._max_pending)
        tolerance_sec = 0.5 / self._sampling_rate if self._sampling_rate > 0 else 1e-6
        while not self._stop.is_set():
            data, timestamps = self._inlet_device.pull_chunk(max_samples=self._pull_size, timeout=0.02)
            pending_device.extend(zip(timestamps, data))
            data, timestamps = self._inlet_twin.pull_chunk(max_samples=self._pull_size, timeout=0.02)
            pending_twin.extend(zip(timestamps, data))
            errors, error_timestamps, num_skipped = self.match(
                pending_device, pending_twin, tolerance_sec
            )
            self._num_skipped += num_skipped
            if errors:
                self._outlet.push_chunk(errors, error_timestamps)
                self._sum_abs_error += int(np.sum(np.abs(errors)))
                self._num_samples += len(errors)
                self._max_abs_error = max(self._max_abs_error, int(np.max(np.abs(errors))))
