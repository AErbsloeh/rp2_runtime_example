from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from logging import getLogger
from threading import Event, Lock, Thread
from time import monotonic

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore

from api.src._live_visualize import XZoomViewBox, get_time_ticks

from .digital_twin import DigitalTwin


@dataclass(frozen=True)
class ComparisonResult:
    """Result of the offline comparison between the twin and the device
    Attributes:
        samples:    Array with the samples sent to the device and the twin
        twin:       Array with the output of the twin
        device:     Array with the output of the device
    """

    samples: np.ndarray
    twin: np.ndarray
    device: np.ndarray

    @property
    def error(self) -> np.ndarray:
        """Error of each sample (device minus twin)"""
        return self.device.astype(np.int64) - self.twin.astype(np.int64)

    @property
    def max_abs_error(self) -> int:
        """Largest absolute error"""
        return int(np.max(np.abs(self.error)))

    def num_mismatch(self, tolerance: int = 0) -> int:
        """Number of samples with an absolute error above the tolerance
        :param tolerance:   Largest accepted absolute error
        :return:            Integer with number of samples
        """
        return int(np.count_nonzero(np.abs(self.error) > tolerance))

    def is_equal(self, tolerance: int = 0) -> bool:
        """Checking if all samples are equal within the tolerance
        :param tolerance:   Largest accepted absolute error
        :return:            True if no sample is above the tolerance
        """
        return self.num_mismatch(tolerance) == 0


def run_on_device(
    process: Callable[[int], None],
    read_results: Callable[[], list[list[int]]],
    samples: np.ndarray,
    timeout_sec: float = 2.0,
    discard_pending: bool = True,
) -> np.ndarray:
    """Sending the samples one by one to the device and collecting the results (as fast as possible, without real time)
    Results of earlier calls waiting in the device are discarded first. The device sends its results block by block, so the number of
    samples must be a multiple of the block size of the output stream (`num_samples` of `daq_out` in the firmware), otherwise the last block stays in the device
    :param process:         Function with one integer sample as input (e.g. `DeviceAPI().process_data`)
    :param read_results:    Function which returns the results as [samples][channels] (e.g. `DeviceAPI().read_processing_results`)
    :param samples:         Array with the integer samples (one channel)
    :param timeout_sec:     Time in seconds without new results until the device is declared as not responding
    :param discard_pending: True to discard the results waiting in the device before sending (False for the following blocks of a continuous run)
    :return:                Array with one result per sample
    """
    num_samples = len(samples)
    while discard_pending and read_results():
        pass
    for sample in samples:
        process(int(sample))

    results: list[int] = []
    deadline = monotonic() + timeout_sec
    while len(results) < num_samples and monotonic() < deadline:
        rows = read_results()
        if rows:
            results.extend(int(row[0]) for row in rows)
            deadline = monotonic() + timeout_sec
    if len(results) < num_samples:
        raise TimeoutError(
            f"The device returned {len(results)} of {num_samples} results, "
            "check if the number of samples is a multiple of the block size of the output stream"
        )
    if len(results) > num_samples:
        raise ValueError(f"The device returned {len(results)} results for {num_samples} samples")
    return np.asarray(results)


def compare_offline(
    twin: DigitalTwin,
    process: Callable[[int], None],
    read_results: Callable[[], list[list[int]]],
    samples: np.ndarray,
    timeout_sec: float = 2.0,
) -> ComparisonResult:
    """Comparing the output of the twin with the output of the device for the same samples without real time and without LSL
    :param twin:            Digital twin with one channel (it is not started, `process` is used)
    :param process:         Function with one integer sample as input (e.g. `DeviceAPI().process_data`)
    :param read_results:    Function which returns the results as [samples][channels] (e.g. `DeviceAPI().read_processing_results`)
    :param samples:         Array with the integer samples (one channel)
    :param timeout_sec:     Time in seconds without new results until the device is declared as not responding
    :return:                Result with the samples, the output of the twin and of the device
    """
    samples = np.asarray(samples).astype(np.int64).reshape(-1)
    twin_out = twin.process(samples)
    if twin_out.shape[1] != 1:
        raise ValueError("Only the pipelines with one channel can be compared with the device")
    device_out = run_on_device(process, read_results, samples, timeout_sec)
    return ComparisonResult(samples=samples, twin=twin_out[:, 0], device=device_out)


class ComparisonPlot:
    """Window with three plots: the input, the output of the twin and the device, and the error (device minus twin)
    With `window_sec` the plots show the last seconds of the data like the live plot of the API: the time axis is relative to the newest sample (-window_sec ... 0).
    The range selected with the left mouse button (or the wheel) sets the width of the shown time span, the plots show the last seconds of this width from then on.
    A double click resets the width. Without `window_sec` the plots show all data of the last call of `set_data` with the time of the signal (e.g. for a finished run)

    Args:
        title (str, optional): Title of the window. Defaults to "Twin vs. device"
        window_sec (float | None, optional): Width of the shown time span in seconds. Defaults to None
    """

    X_PADDING: float = 0.0

    def __init__(self, title: str = "Twin vs. device", window_sec: float | None = None) -> None:
        self._window_sec = window_sec
        self._width_sec = window_sec
        self.window = pg.GraphicsLayoutWidget(title=title)
        self._plots = [
            self.window.addPlot(row=idx, col=0, title=plot_title, viewBox=XZoomViewBox())
            for idx, plot_title in enumerate(("Input", "Output", "Error (device - twin)"))
        ]
        plot_input, plot_output, plot_error = self._plots
        for plot_item in self._plots:
            plot_item.showGrid(x=True, y=True, alpha=0.3)
            plot_item.setMenuEnabled(False)
            plot_item.getViewBox().setAutoVisible(y=True)
            plot_item.getViewBox().sigRangeChangedManually.connect(self._on_zoomed)
        plot_output.addLegend()
        plot_error.setLabel(
            "bottom",
            "Time [s]" if window_sec is not None else "Time",
            units=None if window_sec is not None else "s",
        )
        plot_output.setXLink(plot_input)
        plot_error.setXLink(plot_input)
        self.window.scene().sigMouseClicked.connect(self._on_mouse_clicked)
        self._curve_input = plot_input.plot(pen=pg.mkPen("g", width=2))
        self._curve_twin = plot_output.plot(pen=pg.mkPen("b", width=2), name="Twin")
        self._curve_device = plot_output.plot(pen=pg.mkPen("r", width=1), name="Device")
        self._curve_error = plot_error.plot(pen=pg.mkPen("m", width=2))
        self._has_data = False
        self._set_ticks()

    @property
    def width_sec(self) -> float | None:
        """Width of the shown time span in seconds (None: all data)"""
        return self._width_sec

    def reset_zoom(self) -> None:
        """Show the whole time window again"""
        self._width_sec = self._window_sec
        self._set_ticks()
        self._follow()

    def set_data(
        self,
        time_axis: np.ndarray,
        samples: np.ndarray,
        twin: np.ndarray,
        device: np.ndarray,
        error: np.ndarray,
    ) -> None:
        """Updating the curves, with `window_sec` the time axis is relative to the newest sample
        :param time_axis:   Array with the time of each sample in seconds
        :param samples:     Array with the input samples
        :param twin:        Array with the output of the twin
        :param device:      Array with the output of the device
        :param error:       Array with the error (device minus twin)
        :return:            None
        """
        if self._window_sec is not None and len(time_axis):
            time_axis = time_axis - time_axis[-1]
        self._curve_input.setData(time_axis, samples)
        self._curve_twin.setData(time_axis, twin)
        self._curve_device.setData(time_axis, device)
        self._curve_error.setData(time_axis, error)
        self._has_data = len(time_axis) > 0
        self._follow()

    def _follow(self) -> None:
        if self._width_sec is None or not self._has_data:
            return
        self._plots[0].setXRange(-self._width_sec, 0.0, padding=self.X_PADDING)

    def _set_ticks(self) -> None:
        if self._window_sec is None:
            return
        for plot_item in self._plots:
            plot_item.getAxis("bottom").setTicks(
                None if self._width_sec != self._window_sec else [get_time_ticks(self._window_sec)]
            )

    def _on_zoomed(self, *_) -> None:
        x_min, x_max = self._plots[0].viewRange()[0]
        if self._window_sec is not None and x_max > x_min:
            self._width_sec = x_max - x_min
            self._set_ticks()
            self._follow()

    def _on_mouse_clicked(self, event) -> None:
        if event.button() == QtCore.Qt.MouseButton.LeftButton and event.double():
            self.reset_zoom()
            event.accept()


def build_comparison_window(
    result: ComparisonResult, sampling_rate: float, title: str = "Twin vs. device"
) -> pg.GraphicsLayoutWidget:
    """Building the plot window with the input, the output of the twin and the device and the error (device minus twin)
    :param result:          Result of the comparison
    :param sampling_rate:   Sampling rate in Hz for the time axis
    :param title:           Title of the window
    :return:                Window (not shown)
    """
    view = ComparisonPlot(title)
    time_axis = np.arange(len(result.samples)) / sampling_rate
    view.set_data(time_axis, result.samples, result.twin, result.device, result.error)
    return view.window


def plot_comparison(
    result: ComparisonResult, sampling_rate: float, title: str = "Twin vs. device"
) -> None:
    """Showing the plot of the comparison, this call blocks until the window is closed
    :param result:          Result of the comparison
    :param sampling_rate:   Sampling rate in Hz for the time axis
    :param title:           Title of the window
    :return:                None
    """
    app = pg.mkQApp()
    window = build_comparison_window(result, sampling_rate, title)
    window.show()
    app.exec()


class ContinuousComparison:
    """Comparing the twin with the device continuously without real time: block after block of the source is processed by the twin and sent to the device as fast as the device answers, until it is stopped
    The time axis of the plot is the time of the signal (sample index divided by the sampling rate), not the wall clock, because the speed depends on the device

    Args:
        twin (DigitalTwin): Digital twin with one channel (it is not started, `process` is used)
        process (Callable): Function with one integer sample as input (e.g. `DeviceAPI().process_data`)
        read_results (Callable): Function which returns the results as [samples][channels] (e.g. `DeviceAPI().read_processing_results`)
        source (Callable): Function with the time in seconds (np.ndarray) as input and the integer samples of one channel as output, called for each block
        sampling_rate (float): Sampling rate in Hz of the signal
        block_samples (int, optional): Number of samples sent to the device before its results are collected, must be a multiple of the block size of the output stream (`num_samples` of `daq_out` in the firmware). Defaults to 64
        window_sec (float, optional): Length of the time window of the plot in seconds of the signal. Defaults to 10.0
        timeout_sec (float, optional): Time in seconds without new results until the device is declared as not responding. Defaults to 2.0
    """

    def __init__(
        self,
        twin: DigitalTwin,
        process: Callable[[int], None],
        read_results: Callable[[], list[list[int]]],
        source: Callable[[np.ndarray], np.ndarray],
        sampling_rate: float,
        block_samples: int = 64,
        window_sec: float = 10.0,
        timeout_sec: float = 2.0,
    ) -> None:
        self._twin = twin
        self._process = process
        self._read_results = read_results
        self._source = source
        self._sampling_rate = sampling_rate
        self._block_samples = block_samples
        self._timeout_sec = timeout_sec
        self._window_sec = window_sec
        self._blocks: deque = deque(
            maxlen=max(1, int(np.ceil(window_sec * sampling_rate / block_samples)))
        )
        self._lock = Lock()
        self._stop = Event()
        self._thread: Thread | None = None
        self._exception: Exception | None = None
        self._num_samples = 0
        self._num_mismatch = 0
        self._max_abs_error = 0
        self._sum_abs_error = 0
        self._time_start = monotonic()
        self._logger = getLogger(__name__)

    @property
    def mae(self) -> float:
        """Mean absolute error of all compared samples since the start (0 without samples)"""
        with self._lock:
            return self._sum_abs_error / self._num_samples if self._num_samples else 0.0

    @property
    def num_samples(self) -> int:
        """Number of compared samples since the start"""
        return self._num_samples

    @property
    def num_mismatch(self) -> int:
        """Number of samples with an error since the start"""
        return self._num_mismatch

    @property
    def max_abs_error(self) -> int:
        """Largest absolute error since the start"""
        return self._max_abs_error

    def start(self) -> None:
        """Start the comparison in a thread"""
        if self._thread is not None:
            return
        self._stop.clear()
        self._exception = None
        self._time_start = monotonic()
        self._thread = Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the comparison, log the result and raise the error of the thread (e.g. the device does not respond)"""
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
            self._logger.info(
                f"Continuous comparison: {self._num_samples} samples, {self._num_mismatch} differ from the twin, "
                f"MAE {self.mae:.3f}, max. abs. error {self._max_abs_error}"
            )
        if self._exception is not None:
            raise self._exception

    def snapshot(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
        """Data of the time window
        :return:    None without data, otherwise the arrays time in seconds, input samples, output of the twin, output of the device and error
        """
        with self._lock:
            blocks = list(self._blocks)
        if not blocks:
            return None
        start = np.concatenate(
            [np.arange(first, first + len(samples)) for first, samples, _, _ in blocks]
        )
        samples = np.concatenate([block[1] for block in blocks])
        twin = np.concatenate([block[2] for block in blocks])
        device = np.concatenate([block[3] for block in blocks])
        return (
            start / self._sampling_rate,
            samples,
            twin,
            device,
            device.astype(np.int64) - twin.astype(np.int64),
        )

    def plot(self, title: str = "Twin vs. device", update_interval_ms: int = 50) -> None:
        """Run the comparison and show the plot until the window is closed. The error of the thread is raised afterwards
        :param title:               Title of the window
        :param update_interval_ms:  Time in milliseconds between two updates of the plot
        :return:                    None
        """
        app = pg.mkQApp()
        view = ComparisonPlot(title, window_sec=self._window_sec)
        timer = QtCore.QTimer()
        timer.timeout.connect(lambda: self.refresh(view, title))
        timer.start(update_interval_ms)
        view.window.resize(1200, 800)
        view.window.show()
        self.start()
        app.exec()
        timer.stop()
        self.stop()

    def refresh(self, view: ComparisonPlot, title: str = "Twin vs. device") -> None:
        """Updating the plot with the data of the time window, the window is closed if the comparison failed
        :param view:    Plot to update
        :param title:   Title of the window, extended by the runtime and the running mean absolute error (MAE)
        :return:        None
        """
        if self._exception is not None:
            view.window.close()
            return
        data = self.snapshot()
        if data is not None:
            view.set_data(*data)
            hours, remainder = divmod(int(monotonic() - self._time_start), 3600)
            minutes, seconds = divmod(remainder, 60)
            view.window.setWindowTitle(
                f"{title}\t Runtime: {hours:02d}:{minutes:02d}:{seconds:02d}\t MAE: {self.mae:.2f}"
            )

    def _run(self) -> None:
        try:
            discard_pending = True
            first_sample = 0
            while not self._stop.is_set():
                indices = np.arange(first_sample, first_sample + self._block_samples)
                samples = (
                    np.asarray(self._source(indices / self._sampling_rate)).astype(np.int64).reshape(-1)
                )
                twin_out = self._twin.process(samples)
                if twin_out.shape[1] != 1:
                    raise ValueError(
                        "Only the pipelines with one channel can be compared with the device"
                    )
                device_out = run_on_device(
                    self._process, self._read_results, samples, self._timeout_sec, discard_pending
                )
                discard_pending = False
                error = device_out.astype(np.int64) - twin_out[:, 0].astype(np.int64)
                with self._lock:
                    self._blocks.append((first_sample, samples, twin_out[:, 0], device_out))
                    self._num_samples += len(samples)
                    self._num_mismatch += int(np.count_nonzero(error))
                    self._max_abs_error = max(self._max_abs_error, int(np.max(np.abs(error))))
                    self._sum_abs_error += int(np.sum(np.abs(error)))
                first_sample += self._block_samples
        except Exception as exception:
            self._exception = exception
