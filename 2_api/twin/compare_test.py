import sys
from time import monotonic, sleep

import numpy as np
import pytest

from api.src._helper import get_path_to_project
from twin import (
    ComparisonPlot,
    ContinuousComparison,
    DigitalTwin,
    DigitalTwinConfig,
)
from twin.compare import (
    build_comparison_window,
    compare_offline,
    run_on_device,
)

sys.path.insert(0, str(get_path_to_project("examples")))
from twin_config import FirmwareProcessing  # noqa: E402


class FakeDevice:
    """Device with the firmware of the template: results are sent in blocks, the value is processed byte by byte"""

    def __init__(self, block_size: int = 4, process_value=None, left_over: list | None = None) -> None:
        self._block_size = block_size
        self._process_value = process_value or (
            lambda value: int(FirmwareProcessing.run_sequential(np.array([value]))[0])
        )
        self._pending: list[int] = list(left_over or [])
        self._results: list[int] = []

    def process(self, value: int) -> None:
        self._results.append(self._process_value(value))

    def read_results(self) -> list[list[int]]:
        sleep(0.001)
        num_blocks = len(self._results) // self._block_size
        out = [[value] for value in self._pending + self._results[: num_blocks * self._block_size]]
        self._results = self._results[num_blocks * self._block_size :]
        self._pending = []
        return out


def _twin() -> DigitalTwin:
    return DigitalTwin(
        DigitalTwinConfig(
            pipeline=[FirmwareProcessing.run_sequential],
            lsl_layer_name_datasource="Source",
            save_h5=False,
            num_samples=3,
        )
    )


def test_run_on_device_collects_the_results_of_all_blocks():
    device = FakeDevice(block_size=4)

    results = run_on_device(device.process, device.read_results, np.arange(8), timeout_sec=0.2)

    assert np.array_equal(results, np.arange(8) >> 1)


def test_run_on_device_raises_if_the_last_block_is_not_complete():
    device = FakeDevice(block_size=4)

    with pytest.raises(TimeoutError, match="multiple of the block size"):
        run_on_device(device.process, device.read_results, np.arange(6), timeout_sec=0.1)


def test_run_on_device_discards_results_of_earlier_calls_and_rejects_surplus():
    device = FakeDevice(block_size=4, left_over=[999] * 4)
    results = run_on_device(device.process, device.read_results, np.arange(4), timeout_sec=0.2)
    assert len(results) == 4
    assert 999 not in results

    device = FakeDevice(block_size=4)
    device.read_results = lambda: [[1]] * 5 if device._results else []
    with pytest.raises(ValueError):
        run_on_device(device.process, device.read_results, np.arange(4), timeout_sec=0.1)


def test_compare_offline_finds_no_error_if_twin_and_device_do_the_same():
    device = FakeDevice()
    samples = np.array([32768 + 4000 + 10000 * np.sin(0.3 * idx) for idx in range(12)]).astype(np.int32)

    result = compare_offline(_twin(), device.process, device.read_results, samples, timeout_sec=0.2)

    assert result.is_equal()
    assert result.max_abs_error == 0
    assert np.array_equal(result.error, np.zeros(12))
    assert np.array_equal(result.samples, samples)


def test_compare_offline_reports_the_running_error_of_a_wrong_device():
    device = FakeDevice(process_value=lambda value: (value >> 1) + (3 if value % 2 else 0))
    samples = np.arange(100, 108)

    result = compare_offline(_twin(), device.process, device.read_results, samples, timeout_sec=0.2)

    assert not result.is_equal()
    assert result.num_mismatch() > 0
    assert result.is_equal(tolerance=result.max_abs_error)
    assert np.array_equal(result.error, result.device - result.twin)


@pytest.mark.plot
def test_comparison_window_has_three_plots(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    device = FakeDevice()
    result = compare_offline(_twin(), device.process, device.read_results, np.arange(8), timeout_sec=0.2)

    window = build_comparison_window(result, 500.0)

    assert len(window.ci.items) == 3


def _wait_until(condition, timeout: float = 5.0) -> None:
    deadline = monotonic() + timeout
    while not condition():
        assert monotonic() < deadline, "timeout"
        sleep(0.005)


def _source(t: np.ndarray) -> np.ndarray:
    return 32768 + 10000 * np.sin(2 * np.pi * 5 * t)


def _continuous(
    device: FakeDevice, window_sec: float = 10.0, timeout_sec: float = 0.2
) -> ContinuousComparison:
    return ContinuousComparison(
        twin=_twin(),
        process=device.process,
        read_results=device.read_results,
        source=_source,
        sampling_rate=500.0,
        block_samples=8,
        window_sec=window_sec,
        timeout_sec=timeout_sec,
    )


def test_continuous_comparison_runs_until_it_is_stopped():
    comparison = _continuous(FakeDevice(block_size=4))

    comparison.start()
    _wait_until(lambda: comparison.num_samples >= 80)
    comparison.stop()

    assert comparison.num_samples > 80
    assert comparison.num_samples % 8 == 0
    assert comparison.num_mismatch == 0
    assert comparison.max_abs_error == 0


def test_continuous_comparison_continues_the_signal_and_keeps_only_the_time_window():
    comparison = _continuous(FakeDevice(block_size=4), window_sec=0.1)

    comparison.start()
    _wait_until(lambda: comparison.num_samples >= 200)
    comparison.stop()
    time_axis, samples, twin, device, error = comparison.snapshot()

    assert len(time_axis) <= 56
    assert np.allclose(np.diff(time_axis), 1 / 500.0)
    assert np.array_equal(samples, _source(time_axis).astype(np.int64))
    assert np.array_equal(twin, FirmwareProcessing.run_sequential(samples))
    assert np.array_equal(error, device - twin)


def test_continuous_comparison_counts_the_errors_of_the_device():
    comparison = _continuous(FakeDevice(process_value=lambda value: (value >> 1) + 2))

    comparison.start()
    _wait_until(lambda: comparison.num_samples >= 40)
    comparison.stop()

    assert comparison.num_mismatch > 0
    assert comparison.max_abs_error >= 1


def test_continuous_comparison_raises_the_error_of_the_thread_when_it_is_stopped():
    device = FakeDevice()
    device.read_results = lambda: []
    comparison = _continuous(device, timeout_sec=0.05)

    comparison.start()
    _wait_until(lambda: comparison._exception is not None)

    with pytest.raises(TimeoutError):
        comparison.stop()


def test_continuous_comparison_has_no_data_before_the_start():
    assert _continuous(FakeDevice()).snapshot() is None


@pytest.mark.plot
def test_refresh_updates_the_plot_and_closes_it_after_an_error(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    comparison = _continuous(FakeDevice(block_size=4))
    view = ComparisonPlot(window_sec=10.0)

    comparison.start()
    _wait_until(lambda: comparison.num_samples >= 16)
    comparison.stop()
    comparison.refresh(view, "Test")

    title = view.window.windowTitle()
    assert title.startswith("Test")
    assert "Runtime: 00:00:0" in title
    assert "MAE: 0.00" in title
    assert "samples" not in title
    assert "max" not in title

    comparison._exception = RuntimeError("failed")
    view.window.show()
    comparison.refresh(view)
    assert not view.window.isVisible()


def test_continuous_comparison_has_the_running_mean_absolute_error():
    comparison = _continuous(
        FakeDevice(
            process_value=lambda value: int(FirmwareProcessing.run_sequential(np.array([value]))[0]) + 2
        )
    )
    assert comparison.mae == 0.0

    comparison.start()
    _wait_until(lambda: comparison.num_samples >= 40)
    comparison.stop()

    assert comparison.mae == pytest.approx(2.0)


@pytest.mark.plot
def test_plot_shows_the_last_seconds_relative_to_the_newest_sample_and_keeps_the_zoomed_width(
    monkeypatch,
):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    view = ComparisonPlot(window_sec=4.0)
    time_axis = np.arange(100.0, 106.0, 0.01)
    view.set_data(time_axis, time_axis, time_axis, time_axis, 0 * time_axis)

    assert view._plots[0].viewRange()[0] == pytest.approx([-4.0, 0.0])
    assert view._plots[0].listDataItems()[0].getData()[0][-1] == pytest.approx(0.0)
    assert view._plots[0].listDataItems()[0].getData()[0][0] == pytest.approx(-5.99)

    view._plots[0].setXRange(-3.0, -1.0, padding=0)
    view._on_zoomed()
    assert view.width_sec == pytest.approx(2.0)

    time_axis = np.arange(101.0, 108.0, 0.01)
    view.set_data(time_axis, time_axis, time_axis, time_axis, 0 * time_axis)
    assert view._plots[0].viewRange()[0] == pytest.approx([-2.0, 0.0])
    assert view._plots[2].viewRange()[0] == pytest.approx([-2.0, 0.0])

    view.reset_zoom()
    assert view._plots[0].viewRange()[0] == pytest.approx([-4.0, 0.0])


@pytest.mark.plot
def test_plot_without_window_shows_all_data_and_does_not_follow(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    view = ComparisonPlot()

    view.set_data(np.arange(5.0), np.arange(5), np.arange(5), np.arange(5), np.zeros(5))

    assert view.width_sec is None
