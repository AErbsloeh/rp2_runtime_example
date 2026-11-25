import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pyqtgraph as pg
from pylsl import StreamInlet, proc_threadsafe, resolve_bypred
from pyqtgraph.Qt import QtCore


@dataclass
class ChannelConfig:
    lsl_stream_name: str
    channel_name: str
    channel_unit: str
    curve_color: str
    position: tuple[int, int]
    stream_channel_idx: int = 0


@dataclass
class LivePlotterChannelConfig:
    plot_title: list[list[str|None]] #[row][col]
    channel_configs: list[ChannelConfig]

CURVE_COLORS = ("r", "b", "g", "m", "c", "y")


def build_comparison_plot_config(
    rows: list[tuple[str, str, str]], num_channels: int
) -> LivePlotterChannelConfig:
    """Plot configuration for comparing LSL streams with the same channels: one row per stream, one column per channel
    :param rows:            List with tuples (LSL stream name, label of the row, unit) for each row
    :param num_channels:    Integer with number of channels to show for each stream (stream channels 0 ... N-1)
    :return:                Configuration for the LivePlotter
    """
    if not rows:
        raise ValueError("At least one stream is needed for plotting")
    if num_channels < 1:
        raise ValueError("At least one channel is needed for plotting")
    labels = [label for _, label, _ in rows]
    if len(set(labels)) != len(labels):
        raise ValueError("Labels of the rows must differ, because curves are identified by their name")

    plot_title = [[f"{label} CH{idx}" for idx in range(num_channels)] for label in labels]
    channel_configs = [
        ChannelConfig(
            lsl_stream_name=stream,
            channel_name=f"{label} CH{idx}",
            channel_unit=unit,
            curve_color=CURVE_COLORS[idx % len(CURVE_COLORS)],
            position=(row, idx),
            stream_channel_idx=idx,
        )
        for row, (stream, label, unit) in enumerate(rows)
        for idx in range(num_channels)
    ]
    return LivePlotterChannelConfig(plot_title=plot_title, channel_configs=channel_configs)


def get_time_ticks(window_sec: float, max_intervals: int = 6) -> list[tuple[float, str]]:
    """Ticks for the time axis (time relative to now) incl. start (-window_sec) and end (0) of the window"""
    magnitude = 10 ** math.floor(math.log10(window_sec / max_intervals))
    step = next(factor * magnitude for factor in (1, 2, 5, 10) if window_sec / (factor * magnitude) <= max_intervals)
    values = [idx * step for idx in range(int(window_sec / step + 1e-9) + 1)]
    if len(values) > 1 and window_sec - values[-1] <= 0.5 * step:
        values.pop()
    values.append(window_sec)
    return [(-val, f"{-round(val, 6) + 0.0:g}") for val in reversed(values)]


class RingBuffer:
    def __init__(self, size: int) -> None:
        self._size = size
        self._data = np.zeros(2 * size)
        self._ptr = 0
        self._filled = 0

    def extend(self, values: np.ndarray) -> None:
        values = np.asarray(values)[-self._size:]
        n = len(values)
        idx = (self._ptr + np.arange(n)) % self._size
        self._data[idx] = values
        self._data[idx + self._size] = values
        self._ptr = (self._ptr + n) % self._size
        self._filled = min(self._filled + n, self._size)

    def get(self) -> np.ndarray:
        if self._filled < self._size:
            return self._data[:self._filled]
        return self._data[self._ptr:self._ptr + self._size]

    def __len__(self) -> int:
        return self._filled


class LivePlotter:
    UPDATE_INTERVAL_MS: int = 33
    LINE_WIDTH: int = 2

    def __init__(self, channel_config: LivePlotterChannelConfig, window_width_sec: float| int):
        self.channel_config = channel_config
        self._window_width_sec = window_width_sec
        self.channels = channel_config.channel_configs
        self.inlets = self._connect_lsl_streams()
        self.inlet_by_name = {
            inlet.info().name(): inlet for inlet in self.inlets
        }

        self.max_samples = max(1, int(round(max(inlet.info().nominal_srate()*self._window_width_sec for inlet in self.inlets))))
        self._sample_period_sec = {
            inlet.info().name(): self._get_sample_period(inlet.info().nominal_srate()) for inlet in self.inlets
        }
        self.data_buffers = [RingBuffer(self.max_samples) for _ in self.channels]
        self._time_buffer = RingBuffer(self.max_samples)
        self._start_time = time.monotonic()
        self._last_runtime_sec = -1

        self._init_ui()
        self.timer = self._init_timer()

    def _get_sample_period(self, sampling_rate: float) -> float:
        if sampling_rate > 0:
            return 1.0 / sampling_rate
        return self._window_width_sec / self.max_samples

    def _search_lsl_stream_and_connect(self, lsl_name: str, timeout: float = 2.0) -> StreamInlet:
        streams = resolve_bypred(predicate=f"name='{lsl_name}'", timeout=timeout)

        if not streams:
            raise RuntimeError(f"LSL Stream '{lsl_name}' was not found in {timeout}s !")
        return StreamInlet(
            streams[0],
            max_buflen=60,
            max_chunklen=1024,
            recover=True,
            processing_flags=proc_threadsafe,
        )

    def _connect_lsl_streams(self) -> list[StreamInlet|Any]:
        seen_names = set()
        inlets = []

        for ch in self.channel_config.channel_configs:
            if ch.lsl_stream_name not in seen_names:
                seen_names.add(ch.lsl_stream_name)
                inlet = self._search_lsl_stream_and_connect(lsl_name=ch.lsl_stream_name)
                inlets.append(inlet)

        return inlets

    def _init_ui(self) -> None:
        self._win = pg.GraphicsLayoutWidget(show=True, title="LSL Live Plot\t Runtime: 00:00:00")
        self._win.resize(1400, 700)

        self.plots = {}
        self.curves = {}

        channels_by_pos = {}
        for ch in self.channels:
            channels_by_pos.setdefault(ch.position, []).append(ch)

        for row in range(len(self.channel_config.plot_title)):
            for col in range(len(self.channel_config.plot_title[row])):
                plot_title = self.channel_config.plot_title[row][col]
                if plot_title is None or (row, col) not in channels_by_pos:
                    self._win.addViewBox(row=row, col=col)
                else:
                    plot_item = self._win.addPlot(row=row, col=col)
                    plot_item.showGrid(x=True, y=True, alpha=0.3)
                    plot_item.setXRange(-self._window_width_sec, 0, padding=0.04)
                    plot_item.getAxis("bottom").setTicks([get_time_ticks(self._window_width_sec)])

        num_rows = len(self.channel_config.plot_title)
        num_cols = max(len(row) for row in self.channel_config.plot_title)
        self._win.addLabel("Time [s]", row=num_rows, col=0, colspan=num_cols, justify="center")

        for ch in self.channels:
            row, col = ch.position
            plot_item = self._win.getItem(row=row, col=col)

            if plot_item.legend is None:
                plot_item.addLegend()

            #Labeling of each plot by channelgroups
            group = channels_by_pos[(row, col)]
            combined_label = " / ".join(c.channel_name for c in group)
            unique_units = list(dict.fromkeys(c.channel_unit for c in group if c.channel_unit))
            combined_unit = " / ".join(unique_units)

            plot_item.setLabel('left', combined_label, units=combined_unit)

            pen = pg.mkPen(color=ch.curve_color, width=self.LINE_WIDTH)
            curve = plot_item.plot(pen=pen, name=ch.channel_name)

            self.plots[ch.channel_name] = plot_item
            self.curves[ch.channel_name] = curve

    def _init_timer(self) -> QtCore.QTimer:
        timer = QtCore.QTimer()
        timer.timeout.connect(self._update)
        timer.start(self.UPDATE_INTERVAL_MS)
        return timer

    def _update(self) -> None:
        self._update_runtime()
        self._update_plots()

    def _update_runtime(self):
        elapsed = int(time.monotonic() - self._start_time)
        if elapsed == self._last_runtime_sec:
            return
        self._last_runtime_sec = elapsed
        hours, remainder = divmod(elapsed, 3600)
        minutes, seconds = divmod(remainder, 60)
        self._win.setWindowTitle(f"LSL Live Plot\t Runtime: {hours:02d}:{minutes:02d}:{seconds:02d}")

    def _update_plots(self) -> None:
        chunks_by_stream = {}
        for stream_name, inlet in self.inlet_by_name.items():
            chunk, _ = inlet.pull_chunk(timeout=0.0, max_samples=1024)
            if chunk:
                chunks_by_stream[stream_name] = np.array(chunk)

        for idx, ch in enumerate(self.channels):
            stream_name = ch.lsl_stream_name

            if stream_name in chunks_by_stream:
                data = chunks_by_stream[stream_name]

                if data.ndim > 1 and data.shape[1] > ch.stream_channel_idx:
                    values = data[:, ch.stream_channel_idx]
                else:
                    values = data[:, 0] if data.ndim > 1 else data

                self.data_buffers[idx].extend(values)

            y_data = self.data_buffers[idx].get()
            if len(y_data) > 0:
                x_data = (np.arange(len(y_data)) - len(y_data) + 1) * self._sample_period_sec[stream_name]
                self.curves[ch.channel_name].setData(x_data, y_data)
