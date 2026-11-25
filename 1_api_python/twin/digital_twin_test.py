from functools import partial
from queue import Empty, Queue
from time import monotonic, sleep
from unittest.mock import MagicMock

import h5py
import numpy as np
import pylsl
import pytest

from api.src._lsl import ThreadLSL
from twin import DigitalTwin, DigitalTwinConfig
from twin.digital_twin import TWIN_OUTPUT_STREAM


def double_signal(signal: np.ndarray) -> np.ndarray[np.float32]:
    return (2 * signal).astype(np.float32)


def not_annotated(signal):
    return signal


class FakeInfo:
    def __init__(self, num_channels: int, sampling_rate: float) -> None:
        self._num_channels = num_channels
        self._sampling_rate = sampling_rate

    def channel_count(self) -> int:
        return self._num_channels

    def nominal_srate(self) -> float:
        return self._sampling_rate

    def type(self) -> str:
        return "data"

    def channel_format(self) -> int:
        return pylsl.cf_float32

    def get_channel_units(self) -> list[str]:
        return ["LSB"] * self._num_channels

    def get_channel_labels(self) -> list[str]:
        return [f"CH{idx}" for idx in range(self._num_channels)]

    def get_channel_types(self) -> list[str]:
        return [str([idx]) for idx in range(self._num_channels)]


class FakeInlet:
    def __init__(self, num_channels: int, sampling_rate: float) -> None:
        self._info = FakeInfo(num_channels, sampling_rate)
        self._chunks = Queue()
        self.num_pulled = 0

    def info(self) -> FakeInfo:
        return self._info

    def push(self, data: list, timestamps: list) -> None:
        self._chunks.put((data, timestamps))

    def pull_chunk(self, max_samples: int = 1024, timeout: float = 0.0) -> tuple[list, list]:
        try:
            data, timestamps = self._chunks.get(timeout=0.001)
        except Empty:
            return [], []
        self.num_pulled += len(data)
        return data, timestamps


class FakeOutlet:
    def __init__(self, target: FakeInlet) -> None:
        self._target = target

    def push_chunk(self, x: list, timestamp: list, pushthrough: bool = True) -> None:
        self._target.push(x, timestamp)

    def have_consumers(self) -> bool:
        return True


class FakeLSL:
    """Replacing the LSL streams of ThreadLSL: the outlet of the twin feeds directly the inlet of the recording"""

    def __init__(self, num_channels: int = 2, sampling_rate: float = 1000.0) -> None:
        self.source = FakeInlet(num_channels, sampling_rate)
        self.processed = FakeInlet(num_channels, sampling_rate)
        self.outlet_kwargs: dict = dict()

    def establish_inlet(self, name: str, timeout: float = 10.0) -> FakeInlet:
        return self.processed if name == TWIN_OUTPUT_STREAM else self.source

    def establish_outlet(self, **kwargs) -> FakeOutlet:
        self.outlet_kwargs = kwargs
        return FakeOutlet(self.processed)


@pytest.fixture
def fake(monkeypatch) -> FakeLSL:
    fake = FakeLSL()
    monkeypatch.setattr(ThreadLSL, "_establish_lsl_inlet", fake.establish_inlet)
    monkeypatch.setattr(ThreadLSL, "_establish_lsl_outlet", fake.establish_outlet)
    return fake


def _wait_until(condition, timeout: float = 5.0) -> None:
    deadline = monotonic() + timeout
    while not condition():
        assert monotonic() < deadline, "timeout while waiting for the twin"
        sleep(0.01)


def _make_twin(tmp_path, pipeline=None, save_h5: bool = True) -> DigitalTwin:
    return DigitalTwin(
        DigitalTwinConfig(
            pipeline=[double_signal] if pipeline is None else pipeline,
            lsl_layer_name_datasource="Source",
            save_h5=save_h5,
            path2save=tmp_path,
        )
    )


def test_pipeline_needs_functions_and_return_annotation(tmp_path):
    with pytest.raises(ValueError):
        _make_twin(tmp_path, pipeline=[])
    with pytest.raises(TypeError):
        _make_twin(tmp_path, pipeline=[not_annotated])


def test_stream_names(tmp_path):
    twin = _make_twin(tmp_path)

    assert twin.source_stream_name == "Source"
    assert twin.output_stream_name == TWIN_OUTPUT_STREAM == "DigitalTwinOutput"


def test_output_stream_takes_metadata_of_source_and_format_of_pipeline(tmp_path, fake):
    twin = _make_twin(tmp_path, save_h5=False)

    twin.start()
    fake.source.push([[1, 2]], [100.0])
    twin.stop()

    assert fake.outlet_kwargs["lsl_name"] == TWIN_OUTPUT_STREAM
    assert fake.outlet_kwargs["sampling_rate"] == 1000.0
    assert fake.outlet_kwargs["channel_num"] == 2
    assert fake.outlet_kwargs["channel_type"] == pylsl.cf_float32
    assert fake.outlet_kwargs["channel_labels"] == ["CH0", "CH1"]


def test_processed_data_is_recorded_with_original_timestamps(tmp_path, fake):
    twin = _make_twin(tmp_path)
    data = [[1, 10], [2, 20], [3, 30]]
    timestamps = [100.0, 100.001, 100.002]

    twin.start()
    fake.source.push(data, timestamps)
    _wait_until(lambda: fake.processed.num_pulled >= len(data))
    twin.stop()

    files = list(tmp_path.glob("*_DigitalTwinOutput.h5"))
    assert len(files) == 1
    with h5py.File(files[0], "r") as f:
        np.testing.assert_allclose(f["DigitalTwinOutput_data"][:], 2 * np.array(data))
        np.testing.assert_allclose(f["DigitalTwinOutput_time"][:], timestamps)
        assert f["DigitalTwinOutput_data"].attrs["channel_count"] == 2
        assert f.attrs["process_pipeline"] == "double_signal"


def test_pipeline_is_applied_in_order_and_names_support_partial(tmp_path, fake):
    def add_one(signal: np.ndarray) -> np.ndarray:
        return signal + 1

    twin = _make_twin(tmp_path, pipeline=[add_one, partial(np.multiply, 3), double_signal])

    assert twin._process_chunk([[1, 2]]) == [[12.0, 18.0]]
    assert [DigitalTwin._get_function_name(f) for f in twin._config.pipeline] == [
        "add_one",
        "multiply",
        "double_signal",
    ]


def test_plot_uses_channels_of_source_and_twin_stream(tmp_path, monkeypatch):
    twin = _make_twin(tmp_path)
    monkeypatch.setattr("twin.digital_twin.resolve_bypred", lambda **kwargs: [FakeInfo(2, 1000.0)])
    plot = MagicMock()
    monkeypatch.setattr(ThreadLSL, "lsl_plot_stream", plot)

    twin.plot(window_sec=5.0, twin_unit="V")

    config = plot.call_args.kwargs["plot_config"]
    assert plot.call_args.kwargs["window_length"] == 5.0
    assert [(ch.lsl_stream_name, ch.channel_unit, ch.position) for ch in config.channel_configs] == [
        ("Source", "LSB", (0, 0)),
        ("Source", "LSB", (0, 1)),
        (TWIN_OUTPUT_STREAM, "V", (1, 0)),
        (TWIN_OUTPUT_STREAM, "V", (1, 1)),
    ]


def test_plot_adds_extra_streams_between_source_and_twin(tmp_path, monkeypatch):
    twin = _make_twin(tmp_path)
    monkeypatch.setattr("twin.digital_twin.resolve_bypred", lambda **kwargs: [FakeInfo(1, 1000.0)])
    plot = MagicMock()
    monkeypatch.setattr(ThreadLSL, "lsl_plot_stream", plot)

    twin.plot(extra_streams=[("McuEcho", "MCU", "LSB")])

    config = plot.call_args.kwargs["plot_config"]
    assert [(ch.lsl_stream_name, ch.position) for ch in config.channel_configs] == [
        ("Source", (0, 0)),
        ("McuEcho", (1, 0)),
        (TWIN_OUTPUT_STREAM, (2, 0)),
    ]


def test_plot_raises_if_source_stream_is_missing(tmp_path, monkeypatch):
    twin = _make_twin(tmp_path)
    monkeypatch.setattr("twin.digital_twin.resolve_bypred", lambda **kwargs: [])

    with pytest.raises(ValueError):
        twin.plot()
