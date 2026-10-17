import json
import time
from dataclasses import dataclass
from pathlib import Path
import ast

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
from pylsl import StreamInlet, resolve_bypred, proc_threadsafe


@dataclass
class LivePlotterChannelConfig:
    name: str
    unit: str
    visualized_channel: int
    curve_color: str = "b"
    grid_row: int = None
    grid_col: int = None


@dataclass
class PlotSessionConfig:
    curve_color: str
    channels: list[LivePlotterChannelConfig]


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
    PLOTS_PER_ROW: int = 4
    LINE_WIDTH: int = 2

    def __init__(self, lsl_name: str, window_width_sec: float, path2config: Path | None = None) -> None:
        self._window_width_sec = window_width_sec
        self._inlet = self._search_lsl_stream_and_connect(lsl_name)
        info = self._inlet.info()
        if path2config is None:
            config = load_plot_config_from_stream(self._inlet, curve_color='r')
        else:
            config = load_plot_config_from_json(path2config, curve_color='r')

        channels = config.channels
        self._display_mode = "grid" if all(c.grid_row is not None for c in channels) else "filtered"
        self._visualized_channel = [c.visualized_channel for c in channels]
        self._grid_positions = [(c.grid_row, c.grid_col) for c in channels]
        self._grid_units = [c.unit for c in channels]

        self._validate_channels()
        self.max_samples = max(1, int(round(info.nominal_srate() * self._window_width_sec)))
        self.data_buffers = [RingBuffer(self.max_samples) for _ in channels]
        self._time_buffer = RingBuffer(self.max_samples)
        self._start_time = time.monotonic()
        self._last_runtime_sec = -1
        self._app, self._win, self._plot_items, self._curves, self._curve_source_indices = self._init_plot(
            curve_color=config.curve_color,
            curve_labels=[c.name for c in channels]
        )
        self._timer = self._init_timer()

    def _make_plot_item(self, win: pg.GraphicsLayoutWidget, row: int, col: int, unit: str):
        plot_item = win.addPlot(row=row, col=col)
        plot_item.setLabel("left", "Value", units=f" {unit}" if unit else "")
        plot_item.setLabel("bottom", "Time", units="s")
        plot_item.showGrid(x=True, y=True)
        plot_item.setClipToView(True)
        plot_item.setDownsampling(auto=True, mode="peak")
        plot_item.setXRange(-self._window_width_sec, 0, padding=0.02)
        plot_item.enableAutoRange(axis="x", enable=False)
        num_ticks = max(2, int(self._window_width_sec / 2))
        ticks = [(float(v), f"{v:g}") for v in np.linspace(-self._window_width_sec, 0, num_ticks)]
        plot_item.getAxis("bottom").setTicks([ticks])
        return plot_item

    def _make_curve(self, plot_item, curve_color: str, curve_labels: list[str], idx: int):
        plot_item.addLegend()
        pen = pg.mkPen(color=curve_color, width=self.LINE_WIDTH)
        return plot_item.plot(pen=pen, name=curve_labels[idx])

    def _search_lsl_stream_and_connect(self, lsl_name: str) -> StreamInlet:
        streams = resolve_bypred(predicate=f"name='{lsl_name}'")
        if not streams:
            raise RuntimeError(f"No Stream with Layer Name {lsl_name} found!")
        return StreamInlet(
            streams[0],
            max_buflen=60,
            max_chunklen=1024,
            recover=True,
            processing_flags=proc_threadsafe,
        )

    def _validate_channels(self) -> None:
        channel_count = self._inlet.channel_count
        for ch in self._visualized_channel:
            if ch < 0 or ch >= channel_count:
                raise ValueError(
                    f"Visualized channel index {ch} is outside the stream channel range 0..{channel_count - 1}."
                )

    def _init_plot(self, curve_color: str, curve_labels: list[str]) -> tuple:
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        win = pg.GraphicsLayoutWidget(show=True, title="LSL Live Plot")
        win.resize(1400, 700)

        if self._display_mode == "grid":
            num_cols = max((col for _, col in self._grid_positions), default=0) + 1
        else:
            num_cols = max(1, min(len(self.data_buffers), self.PLOTS_PER_ROW))
        self._runtime_label = win.addLabel(
            "Laufzeit: 00:00:00", row=0, col=0, colspan=num_cols, justify="left", size="12pt"
        )

        plot_items = []
        curves = []
        curve_source_indices = []

        if self._display_mode == "grid":
            for idx in range(len(self.data_buffers)):
                row, col = self._grid_positions[idx]
                plot_item = self._make_plot_item(win, row + 1, col, self._grid_units[idx])
                curve = self._make_curve(plot_item, curve_color, curve_labels, idx)
                plot_items.append(plot_item)
                curves.append(curve)
                curve_source_indices.append(idx)
        else:
            for idx in range(len(self.data_buffers)):
                row, col = divmod(idx, self.PLOTS_PER_ROW)
                plot_item = self._make_plot_item(win, row + 1, col, self._grid_units[idx])
                curve = self._make_curve(plot_item, curve_color, curve_labels, idx)
                plot_items.append(plot_item)
                curves.append(curve)
                curve_source_indices.append(idx)

        return app, win, plot_items, curves, curve_source_indices

    def _init_timer(self) -> QtCore.QTimer:
        timer = QtCore.QTimer()
        timer.timeout.connect(self._update)
        timer.start(self.UPDATE_INTERVAL_MS)
        return timer

    def _append_chunk(self, data: np.ndarray, timestamps: np.ndarray) -> None:
        for i, ch in enumerate(self._visualized_channel):
            column = data[:, ch]
            self.data_buffers[i].extend(column)
        self._time_buffer.extend(timestamps)

    def _update_runtime_label(self) -> None:
        elapsed = int(time.monotonic() - self._start_time)
        if elapsed == self._last_runtime_sec:
            return
        self._last_runtime_sec = elapsed
        hours, remainder = divmod(elapsed, 3600)
        minutes, seconds = divmod(remainder, 60)
        self._runtime_label.setText(f"Laufzeit: {hours:02d}:{minutes:02d}:{seconds:02d}")

    def _update(self) -> None:
        self._update_runtime_label()

        data, timestamps = self._inlet.pull_chunk(timeout=0.0, max_samples=self.max_samples)
        if len(timestamps) > 0:
            self._append_chunk(np.asarray(data), np.asarray(timestamps))

        if len(self._time_buffer) == 0:
            return

        time_axis = self._time_buffer.get()
        time_axis = time_axis - time_axis[-1]

        for curve, src_idx in zip(self._curves, self._curve_source_indices):
            if curve is None:
                continue
            curve.setData(time_axis, self.data_buffers[src_idx].get(), skipFiniteCheck=True)

    def start(self) -> None:
        QtWidgets.QApplication.instance().exec()


def load_plot_config_from_json(config_path: Path, curve_color: str) -> PlotSessionConfig:
    with config_path.open("r", encoding="utf-8") as config_file:
        config_data = json.load(config_file)

    channels = config_data.get("channels")
    if not isinstance(channels, list):
        raise ValueError('The plot configuration must contain a "channels" list.')
    display_mode = config_data.get("display_mode", "filtered")
    grid_layout = config_data.get("grid_layout") or []

    position_by_channel_number = {
        channel_number: (row_idx, col_idx)
        for row_idx, row in enumerate(grid_layout)
        for col_idx, channel_number in enumerate(row)
        if channel_number
    }
    if display_mode != "grid":
        position_by_channel_number = {}

    plot_configs: list[LivePlotterChannelConfig] = []
    for channel in channels:
        channel_number = channel["channel_number"]
        grid_row, grid_col = position_by_channel_number.get(channel_number, (None, None))
        plot_configs.append(
            LivePlotterChannelConfig(
                name=channel["name"],
                visualized_channel=channel_number - 1,
                curve_color=channel.get("curve_color", "b"),
                grid_row=grid_row,
                grid_col=grid_col,
                unit=channel.get("unit", ""),
            )
        )

    if display_mode == "grid" and any(c.grid_row is None for c in plot_configs):
        raise ValueError('Display mode "grid" requires every channel to appear in "grid_layout".')

    return PlotSessionConfig(
        channels=plot_configs,
        curve_color=curve_color
    )


def load_plot_config_from_stream(inlet: StreamInlet, curve_color: str) -> PlotSessionConfig:
    info = inlet.info()
    channel_count = info.channel_count()
    labels = info.get_channel_labels() or []
    units = info.get_channel_units() or []
    types = info.get_channel_types() or []

    def parse_layout(entry: str | None) -> list[list[int]]:
        if not isinstance(entry, str) or not entry.strip().startswith("["):
            return []
        try:
            rows = json.loads(entry)
        except ValueError:
            rows = ast.literal_eval(entry)
        return [json.loads(row) if isinstance(row, str) else [int(x) for x in row] for row in rows]

    grid_layout = parse_layout(types[0] if types else None)
    position_by_channel_number = {
        int(channel_number): (row_idx, col_idx)
        for row_idx, row in enumerate(grid_layout)
        for col_idx, channel_number in enumerate(row)
        if channel_number
    }

    channels: list[LivePlotterChannelConfig] = []
    for i in range(channel_count):
        grid_row, grid_col = position_by_channel_number.get(i + 1, (None, None))
        channels.append(
            LivePlotterChannelConfig(
                name=labels[i] if i < len(labels) and labels[i] else f"Channel {i + 1}",
                visualized_channel=i,
                curve_color=curve_color,
                grid_row=grid_row,
                grid_col=grid_col,
                unit=units[i] if i < len(units) and units[i] else "",
            )
        )
    return PlotSessionConfig(curve_color=curve_color, channels=channels)
