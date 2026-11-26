from collections import deque
from collections.abc import Callable
from logging import getLogger
from threading import Event, Thread
from time import monotonic, sleep

import numpy as np
from pylsl import StreamInfo, StreamInlet, StreamOutlet, cf_int32, resolve_byprop


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
    """Generating test data in a LSL stream (int32) with the nominal sampling rate, e.g. as source for the DigitalTwin

    Args:
        name (str): Name of the LSL stream
        sampling_rate (float): Sampling rate in Hz
        func (Callable): Function with the time in seconds (np.ndarray) as input and the values as output, 1D for one channel or [samples, channels]
        label (str, optional): Label of the channel. Defaults to "data"
        unit (str, optional): Unit of the channel. Defaults to "LSB"
    """

    def __init__(
        self,
        name: str,
        sampling_rate: float,
        func: Callable[[np.ndarray], np.ndarray],
        label: str = "data",
        unit: str = "LSB",
    ) -> None:
        self._fs = sampling_rate
        self._func = func
        self._stop = Event()
        self._thread: Thread | None = None
        probe = np.asarray(func(np.zeros(1)))
        num_channels = 1 if probe.ndim == 1 else probe.shape[1]
        self._outlet = _create_outlet(name, sampling_rate, num_channels, label, unit, "data")

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None

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
            and not as return value of `process`. The results are assigned to the source samples in the order of sending, so the device must return exactly one result per sample. Defaults to None
        name_in (str): Name of the LSL stream with the data to send
        name_out (str, optional): Name of the LSL stream with the returned data. Defaults to "DeviceLoopback"
        label (str, optional): Label of the returned channel. Defaults to "device"
        unit (str, optional): Unit of the returned channel. Defaults to "LSB"
    """

    def __init__(
        self,
        process: Callable[[int], int | None],
        name_in: str,
        name_out: str = "DeviceLoopback",
        label: str = "device",
        unit: str = "LSB",
        read_results: Callable[[], list[list[int]]] | None = None,
    ) -> None:
        self._process = process
        self._read_results = read_results
        self._pending_timestamps: deque[float] = deque()
        self._num_unassigned = 0
        self._stop = Event()
        self._thread: Thread | None = None
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

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
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
        while not self._stop.is_set():
            data, timestamps = self._inlet.pull_chunk(
                max_samples=16, timeout=0.02 if self._read_results is not None else 0.1
            )
            if not data and self._read_results is None:
                continue
            returned = []
            returned_timestamps = []
            for sample, timestamp in zip(data, timestamps):
                t_start = monotonic()
                result = self._process(int(sample[0]))
                self._round_trip_sec.append(monotonic() - t_start)
                if self._read_results is not None:
                    self._pending_timestamps.append(timestamp)
                    continue
                if result is None:
                    self._num_no_response += 1
                    continue
                returned.append([result])
                returned_timestamps.append(timestamp)
            if self._read_results is not None:
                for row in self._read_results():
                    if not self._pending_timestamps:
                        self._num_unassigned += 1
                        continue
                    returned.append([int(row[0])])
                    returned_timestamps.append(self._pending_timestamps.popleft())
            if returned:
                self._outlet.push_chunk(returned, returned_timestamps)
