from time import monotonic, sleep

import numpy as np
import pytest

from twin import streams
from twin.streams import DeviceLoopback, LslGenerator


class FakeInfo:
    def __init__(self, name="", type="data", channel_count=1, nominal_srate=0.0, channel_format=0, source_id="") -> None:
        self.name = name
        self.type_ = type
        self.num_channels = channel_count
        self.srate = nominal_srate
        self.labels = []
        self.units = []

    def set_channel_labels(self, labels) -> None:
        self.labels = labels

    def set_channel_units(self, units) -> None:
        self.units = units

    def set_channel_types(self, types) -> None:
        pass

    def channel_count(self) -> int:
        return self.num_channels

    def nominal_srate(self) -> float:
        return self.srate

    def type(self) -> str:
        return self.type_


class FakeOutlet:
    def __init__(self, info) -> None:
        self.info = info
        self.chunks = []
        self.timestamps = []

    def push_chunk(self, data, timestamps=None) -> None:
        self.chunks.append(data)
        self.timestamps.append(timestamps)


class FakeInlet:
    chunks = []

    def __init__(self, info, **kwargs) -> None:
        self._info = info

    def info(self):
        return self._info

    def pull_chunk(self, max_samples, timeout):
        if FakeInlet.chunks:
            return FakeInlet.chunks.pop(0)
        sleep(0.01)
        return [], []


def _wait_until(condition, timeout=3.0) -> None:
    start = monotonic()
    while not condition():
        assert monotonic() - start < timeout, "timeout"
        sleep(0.01)


@pytest.fixture(autouse=True)
def fake_lsl(monkeypatch):
    monkeypatch.setattr(streams, "StreamInfo", FakeInfo)
    monkeypatch.setattr(streams, "StreamOutlet", FakeOutlet)
    monkeypatch.setattr(streams, "StreamInlet", FakeInlet)
    FakeInlet.chunks = []


def test_generator_pushes_the_function_values_of_consecutive_samples():
    generator = LslGenerator("Gen", 1000.0, lambda t: 1000 * t, label="DAC", unit="V")
    outlet = generator._outlet
    assert (outlet.info.name, outlet.info.srate, outlet.info.labels, outlet.info.units) == ("Gen", 1000.0, ["DAC"], ["V"])

    generator.start()
    _wait_until(lambda: sum(len(chunk) for chunk in outlet.chunks) >= 20)
    generator.stop()

    values = np.array([sample[0] for chunk in outlet.chunks for sample in chunk])
    assert np.array_equal(values, np.arange(len(values)))


def test_generator_supports_several_channels():
    generator = LslGenerator("Gen", 100.0, lambda t: np.stack([t, 2 * t], axis=1))
    assert generator._outlet.info.num_channels == 2
    assert generator._outlet.info.labels == ["data0", "data1"]


def test_loopback_returns_results_with_timestamps_and_skips_missing_responses(monkeypatch):
    monkeypatch.setattr(streams, "resolve_byprop", lambda *args, **kwargs: [FakeInfo("In", "data", 1, 500.0)])
    FakeInlet.chunks = [([[10], [20], [30]], [1.0, 2.0, 3.0])]

    def process(value: int):
        return None if value == 20 else value + 1

    loopback = DeviceLoopback(process, "In", name_out="Out", label="MCU", unit="V")
    assert (loopback._outlet.info.name, loopback._outlet.info.srate, loopback._outlet.info.labels) == ("Out", 500.0, ["MCU"])

    loopback.start()
    _wait_until(lambda: loopback._outlet.chunks)
    loopback.stop()

    assert loopback._outlet.chunks == [[[11], [31]]]
    assert loopback._outlet.timestamps == [[1.0, 3.0]]
    assert loopback._num_no_response == 1


def test_loopback_raises_if_source_stream_is_missing(monkeypatch):
    monkeypatch.setattr(streams, "resolve_byprop", lambda *args, **kwargs: [])
    with pytest.raises(ValueError):
        DeviceLoopback(lambda value: value, "In")


def test_loopback_rejects_streams_with_several_channels(monkeypatch):
    monkeypatch.setattr(streams, "resolve_byprop", lambda *args, **kwargs: [FakeInfo("In", "data", 2, 500.0)])
    with pytest.raises(ValueError):
        DeviceLoopback(lambda value: value, "In")
