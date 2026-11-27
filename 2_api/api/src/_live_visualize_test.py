import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pyqtgraph as pg
import pytest
from pyqtgraph.Qt import QtCore, QtWidgets

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
    def info(self, timeout: float = 0.0) -> FakeInfo:
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


class FakeClick:
    def __init__(self, button, double: bool = False) -> None:
        self._button = button
        self._double = double
        self.accepted = False

    def button(self):
        return self._button

    def double(self) -> bool:
        return self._double

    def accept(self) -> None:
        self.accepted = True


class FakeDrag:
    def __init__(
        self, x_down: float, x_now: float, finish: bool, button=QtCore.Qt.MouseButton.LeftButton
    ) -> None:
        self._down = QtCore.QPointF(x_down, 10.0)
        self._now = QtCore.QPointF(x_now, 10.0)
        self._finish = finish
        self._button = button

    def button(self):
        return self._button

    def buttonDownPos(self):
        return self._down

    def pos(self):
        return self._now

    def isFinish(self) -> bool:
        return self._finish

    def accept(self) -> None:
        pass


def _make_two_plots(app, monkeypatch) -> LivePlotter:
    monkeypatch.setattr(LivePlotter, "_connect_lsl_streams", lambda self: [FakeInlet(samples=50)])
    config = LivePlotterChannelConfig(
        plot_title=[["plot1"], ["plot2"]],
        channel_configs=[
            ChannelConfig("data", "CH0", "LSB", "r", (0, 0), 0),
            ChannelConfig("data", "CH1", "LSB", "b", (1, 0), 1),
        ],
    )
    dut = LivePlotter(config, window_width_sec=2.0)
    app.processEvents()
    return dut


def _select_range(dut: LivePlotter, start: float, stop: float) -> None:
    """Drag with the left mouse button over the given fractions of the width of the first plot"""
    vb = dut.plots["CH0"].getViewBox()
    width = vb.width()
    vb.mouseDragEvent(FakeDrag(start * width, stop * width, finish=False))
    vb.mouseDragEvent(FakeDrag(start * width, stop * width, finish=True))


def test_selected_time_range_is_shown_in_all_plots_and_freezes(app, monkeypatch):
    dut = _make_two_plots(app, monkeypatch)
    full_start, full_stop = dut.plots["CH0"].viewRange()[0]

    _select_range(dut, 0.25, 0.5)

    for name in ("CH0", "CH1"):
        start, stop = dut.plots[name].viewRange()[0]
        assert (stop - start) == pytest.approx(0.25 * (full_stop - full_start), rel=0.05)
        assert full_start < start < stop < full_stop
    assert dut.plots["CH0"].viewRange()[0] == pytest.approx(dut.plots["CH1"].viewRange()[0])
    assert dut.is_frozen


def test_too_small_selection_is_ignored(app, monkeypatch):
    dut = _make_two_plots(app, monkeypatch)
    before = dut.plots["CH0"].viewRange()[0]

    _select_range(dut, 0.5, 0.5 + 2 / dut.plots["CH0"].getViewBox().width())

    assert dut.plots["CH0"].viewRange()[0] == pytest.approx(before)
    assert not dut.is_frozen


def test_double_click_resets_zoom_and_continues_live(app, monkeypatch):
    dut = _make_two_plots(app, monkeypatch)
    full_range = dut.plots["CH0"].viewRange()[0]
    _select_range(dut, 0.25, 0.5)
    assert dut.is_frozen

    click = FakeClick(QtCore.Qt.MouseButton.LeftButton, double=True)
    dut._win.scene().sigMouseClicked.emit(click)

    assert click.accepted
    assert not dut.is_frozen
    for name in ("CH0", "CH1"):
        assert dut.plots[name].viewRange()[0] == pytest.approx(full_range)


def test_single_left_click_does_nothing(app, monkeypatch):
    dut = _make_two_plots(app, monkeypatch)
    click = FakeClick(QtCore.Qt.MouseButton.LeftButton, double=False)
    dut._win.scene().sigMouseClicked.emit(click)
    assert not click.accepted
    assert not dut.is_frozen


def test_mouse_changes_only_the_time_axis(app, monkeypatch):
    dut = _make_two_plots(app, monkeypatch)
    assert dut.plots["CH0"].getViewBox().state["mouseEnabled"] == [True, False]


def test_title_shows_runtime_and_the_info_of_the_caller(app, monkeypatch):
    monkeypatch.setattr(LivePlotter, "_connect_lsl_streams", lambda self: [FakeInlet(samples=4)])
    config = LivePlotterChannelConfig(
        plot_title=[["plot1"]],
        channel_configs=[ChannelConfig("data", "CH0", "LSB", "r", (0, 0), 0)],
    )

    dut = LivePlotter(config, window_width_sec=2.0, title_info=lambda: "MAE: 1.50")
    dut._update_title()

    assert "Runtime: 00:00:00" in dut._win.windowTitle()
    assert "MAE: 1.50" in dut._win.windowTitle()
    without_info = LivePlotter(config, window_width_sec=2.0)
    without_info._update_title()
    assert "MAE" not in without_info._win.windowTitle()


def test_right_click_freezes_all_plots_and_second_click_continues(app, monkeypatch):
    inlet = FakeInlet(samples=10)
    monkeypatch.setattr(LivePlotter, "_connect_lsl_streams", lambda self: [inlet])
    config = LivePlotterChannelConfig(
        plot_title=[["plot1"]],
        channel_configs=[ChannelConfig("data", "CH0", "LSB", "r", (0, 0), 0)],
    )
    dut = LivePlotter(config, window_width_sec=2.0)
    dut._update_plots()
    assert len(dut.curves["CH0"].getData()[0]) == 10

    left_click = FakeClick(QtCore.Qt.MouseButton.LeftButton)
    dut._win.scene().sigMouseClicked.emit(left_click)
    assert not dut.is_frozen
    assert not left_click.accepted

    right_click = FakeClick(QtCore.Qt.MouseButton.RightButton)
    dut._win.scene().sigMouseClicked.emit(right_click)
    assert dut.is_frozen
    assert right_click.accepted
    assert "FROZEN" in dut._win.windowTitle()

    inlet._samples = 5
    dut._update_plots()
    assert len(dut.curves["CH0"].getData()[0]) == 10
    assert len(dut.data_buffers[0]) == 15

    dut._win.scene().sigMouseClicked.emit(FakeClick(QtCore.Qt.MouseButton.RightButton))
    assert not dut.is_frozen
    assert "FROZEN" not in dut._win.windowTitle()
    dut._update_plots()
    assert len(dut.curves["CH0"].getData()[0]) == 15


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
