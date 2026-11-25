import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pyqtgraph as pg
import pytest
from pyqtgraph.Qt import QtWidgets

from api.src._live_visualize import (
    CURVE_COLORS,
    ChannelConfig,
    LivePlotter,
    LivePlotterChannelConfig,
    build_comparison_plot_config,
    get_time_ticks,
)


class FakeInfo:
    def name(self) -> str:
        return "data"

    def nominal_srate(self) -> float:
        return 100.0


class FakeInlet:
    def info(self) -> FakeInfo:
        return FakeInfo()

    def __init__(self, samples: int = 0) -> None:
        self._samples = samples

    def pull_chunk(self, timeout: float = 0.0, max_samples: int = 1024):
        chunk = [[idx, idx] for idx in range(self._samples)]
        self._samples = 0
        return chunk, [0.0] * len(chunk)


@pytest.fixture
def app():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


def test_plot_cells_without_channels_stay_empty(app, monkeypatch):
    monkeypatch.setattr(LivePlotter, "_connect_lsl_streams", lambda self: [FakeInlet()])
    config = LivePlotterChannelConfig(
        plot_title=[["plot1", "plot2"], [None, "plot3"]],
        channel_configs=[
            ChannelConfig("data", "CH0", "LSB", "r", (0, 0), 0),
            ChannelConfig("data", "CH1", "LSB", "b", (1, 1), 1),
        ],
    )

    dut = LivePlotter(config, window_width_sec=2.0)

    assert isinstance(dut._win.getItem(row=0, col=0), pg.PlotItem)
    assert isinstance(dut._win.getItem(row=1, col=1), pg.PlotItem)
    assert isinstance(dut._win.getItem(row=0, col=1), pg.ViewBox)
    assert isinstance(dut._win.getItem(row=1, col=0), pg.ViewBox)
    assert set(dut.curves) == {"CH0", "CH1"}


def test_plots_use_seconds_on_x_axis_with_one_label_below(app, monkeypatch):
    monkeypatch.setattr(LivePlotter, "_connect_lsl_streams", lambda self: [FakeInlet(samples=50)])
    config = LivePlotterChannelConfig(
        plot_title=[["plot1", "plot2"]],
        channel_configs=[ChannelConfig("data", "CH0", "LSB", "r", (0, 0), 0)],
    )

    dut = LivePlotter(config, window_width_sec=2.0)
    dut._update_plots()

    plot = dut._win.getItem(row=0, col=0)
    assert plot.titleLabel.isVisible() is False
    assert plot.viewRange()[0] == pytest.approx([-2.08, 0.08])
    x_data, y_data = dut.curves["CH0"].getData()
    assert len(x_data) == len(y_data) == 50
    assert x_data[-1] == 0.0
    assert x_data[0] == pytest.approx(-49 / 100.0)
    label = dut._win.getItem(row=1, col=0)
    assert isinstance(label, pg.LabelItem)
    assert label.text == "Time [s]"
    assert dut._win.ci.layout.itemAt(1, 0) is label


@pytest.mark.parametrize(
    ("window", "labels"),
    [
        (10.0, ["-10", "-8", "-6", "-4", "-2", "0"]),
        (4.0, ["-4", "-3", "-2", "-1", "0"]),
        (2.5, ["-2.5", "-2", "-1.5", "-1", "-0.5", "0"]),
        (7.0, ["-7", "-4", "-2", "0"]),
    ],
)
def test_time_ticks_include_start_and_end_of_window(window, labels):
    ticks = get_time_ticks(window)

    assert [label for _, label in ticks] == labels
    assert ticks[0][0] == -window
    assert ticks[-1][0] == 0.0


def test_comparison_config_has_one_row_per_stream_and_one_column_per_channel():
    config = build_comparison_plot_config(
        [("data0", "Input", "LSB"), ("twin", "Twin", "V")], num_channels=2
    )

    assert config.plot_title == [["Input CH0", "Input CH1"], ["Twin CH0", "Twin CH1"]]
    assert [
        (ch.lsl_stream_name, ch.channel_unit, ch.position, ch.stream_channel_idx)
        for ch in config.channel_configs
    ] == [
        ("data0", "LSB", (0, 0), 0),
        ("data0", "LSB", (0, 1), 1),
        ("twin", "V", (1, 0), 0),
        ("twin", "V", (1, 1), 1),
    ]
    names = [ch.channel_name for ch in config.channel_configs]
    assert len(set(names)) == len(names)


def test_comparison_config_colors_repeat_for_many_channels():
    config = build_comparison_plot_config([("data0", "Input", "LSB")], num_channels=len(CURVE_COLORS) + 1)

    assert config.channel_configs[0].curve_color == config.channel_configs[-1].curve_color


def test_comparison_config_rejects_invalid_arguments():
    with pytest.raises(ValueError):
        build_comparison_plot_config([], 1)
    with pytest.raises(ValueError):
        build_comparison_plot_config([("data0", "Input", "LSB")], 0)
    with pytest.raises(ValueError):
        build_comparison_plot_config([("a", "Same", "LSB"), ("b", "Same", "LSB")], 1)
