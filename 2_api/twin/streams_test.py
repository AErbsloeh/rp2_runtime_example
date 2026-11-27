from time import monotonic, sleep

import numpy as np
import pytest

from twin import streams
from twin.streams import DeviceLoopback, LslGenerator


class FakeInfo:
    def __init__(
        self, name="", type="data", channel_count=1, nominal_srate=0.0, channel_format=0, source_id=""
    ) -> None:
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

    def info(self, timeout=0.0):
        return self._info

    def open_stream(self, timeout=0.0):
        pass

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
    assert (outlet.info.name, outlet.info.srate, outlet.info.labels, outlet.info.units) == (
        "Gen",
        1000.0,
        ["DAC"],
        ["V"],
    )

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
    monkeypatch.setattr(
        streams, "resolve_byprop", lambda *args, **kwargs: [FakeInfo("In", "data", 1, 500.0)]
    )
    FakeInlet.chunks = [([[10], [20], [30]], [1.0, 2.0, 3.0])]

    def process(value: int):
        return None if value == 20 else value + 1

    loopback = DeviceLoopback(process, "In", name_out="Out", label="MCU", unit="V")
    assert (loopback._outlet.info.name, loopback._outlet.info.srate, loopback._outlet.info.labels) == (
        "Out",
        500.0,
        ["MCU"],
    )

    loopback.start()
    _wait_until(lambda: loopback._outlet.chunks)
    loopback.stop()

    assert loopback._outlet.chunks == [[[11], [31]]]
    assert loopback._outlet.timestamps == [[1.0, 3.0]]
    assert loopback._num_no_response == 1


def test_loopback_assigns_delayed_results_to_the_timestamps_of_the_source(monkeypatch):
    monkeypatch.setattr(
        streams, "resolve_byprop", lambda *args, **kwargs: [FakeInfo("In", "data", 1, 500.0)]
    )
    FakeInlet.chunks = [([[10], [20], [30]], [1.0, 2.0, 3.0])]
    sent = []
    blocks = [[], [[11], [21]], [], [[31], [99]]]

    def read_results():
        return blocks.pop(0) if blocks else []

    loopback = DeviceLoopback(
        lambda value: sent.append(value), "In", name_out="Out", label="MCU", read_results=read_results
    )
    loopback.start()
    _wait_until(lambda: len(loopback._outlet.chunks) >= 2)
    loopback.stop()

    assert sent == [10, 20, 30]
    assert loopback._outlet.chunks == [[[11], [21]], [[31]]]
    assert loopback._outlet.timestamps == [[1.0, 2.0], [3.0]]
    assert loopback._num_unassigned == 1


def test_loopback_reads_the_results_once_before_it_sends_the_first_sample(monkeypatch):
    monkeypatch.setattr(
        streams, "resolve_byprop", lambda *args, **kwargs: [FakeInfo("In", "data", 1, 500.0)]
    )
    FakeInlet.chunks = [([[10], [20]], [1.0, 2.0])]
    calls = []

    loopback = DeviceLoopback(
        lambda value: calls.append(("process", value)),
        "In",
        read_results=lambda: calls.append(("read", None)) or [],
    )
    loopback.start()
    _wait_until(lambda: len(calls) >= 4)
    loopback.stop()

    assert calls[0] == ("read", None)
    assert [call for call in calls if call[0] == "process"][:2] == [("process", 10), ("process", 20)]
    assert calls.index(("process", 10)) > 0


def test_loopback_raises_if_source_stream_is_missing(monkeypatch):
    monkeypatch.setattr(streams, "resolve_byprop", lambda *args, **kwargs: [])
    with pytest.raises(ValueError):
        DeviceLoopback(lambda value: value, "In")


def test_loopback_rejects_streams_with_several_channels(monkeypatch):
    monkeypatch.setattr(
        streams, "resolve_byprop", lambda *args, **kwargs: [FakeInfo("In", "data", 2, 500.0)]
    )
    with pytest.raises(ValueError):
        DeviceLoopback(lambda value: value, "In")


def test_error_monitor_matches_samples_by_timestamp_and_skips_samples_without_partner():
    from collections import deque

    from twin.streams import ErrorMonitor

    pending_device = deque([(1.0, [11]), (2.0, [20]), (4.0, [43])])
    pending_twin = deque([(0.5, [0]), (1.0, [10]), (2.0, [22]), (3.0, [30]), (4.0, [40]), (5.0, [50])])

    errors, timestamps, num_skipped = ErrorMonitor.match(pending_device, pending_twin, 0.1)

    assert errors == [[1], [-2], [3]]
    assert timestamps == [1.0, 2.0, 4.0]
    assert num_skipped == 2
    assert not pending_device
    assert list(pending_twin) == [(5.0, [50])]


def test_error_monitor_waits_for_the_delayed_stream():
    from collections import deque

    from twin.streams import ErrorMonitor

    pending_device: deque = deque()
    pending_twin = deque([(1.0, [10]), (2.0, [20])])

    assert ErrorMonitor.match(pending_device, pending_twin, 0.1) == ([], [], 0)
    assert len(pending_twin) == 2


def test_error_monitor_publishes_the_error_stream(monkeypatch):
    from twin.streams import ErrorMonitor

    class StreamInlets:
        def __init__(self, info, **kwargs) -> None:
            self.name = info.name
            self.chunks = {
                "Device": [([[11], [21], [31]], [1.0, 2.0, 3.0])],
                "Twin": [([[10], [20]], [1.0, 2.0]), ([[30]], [3.0])],
            }[info.name]

        def info(self, timeout=0.0):
            return FakeInfo(self.name, "data", 1, 500.0)

        def open_stream(self, timeout=0.0):
            pass

        def pull_chunk(self, max_samples, timeout):
            if self.chunks:
                return self.chunks.pop(0)
            sleep(0.005)
            return [], []

    monkeypatch.setattr(
        streams, "resolve_byprop", lambda prop, name, timeout: [FakeInfo(name, "data", 1, 500.0)]
    )
    monkeypatch.setattr(streams, "StreamInlet", StreamInlets)
    monitor = ErrorMonitor("Device", "Twin", name_out="Err", label="Error", unit="V")

    monitor.start()
    _wait_until(lambda: sum(len(chunk) for chunk in monitor._outlet.chunks) >= 3)
    monitor.stop()

    assert (monitor._outlet.info.name, monitor._outlet.info.srate, monitor._outlet.info.labels) == (
        "Err",
        500.0,
        ["Error"],
    )
    assert [sample for chunk in monitor._outlet.chunks for sample in chunk] == [[1], [1], [1]]
    assert [ts for stamps in monitor._outlet.timestamps for ts in stamps] == [1.0, 2.0, 3.0]
    assert monitor.max_abs_error == 1
    assert monitor.mae == pytest.approx(1.0)


def test_error_monitor_raises_if_a_stream_is_missing(monkeypatch):
    from twin.streams import ErrorMonitor

    monkeypatch.setattr(streams, "resolve_byprop", lambda prop, name, timeout: [])

    with pytest.raises(ValueError):
        ErrorMonitor("Device", "Twin").start()


def test_generator_without_realtime_pushes_blocks_on_request_with_the_time_of_the_signal():
    generator = LslGenerator("Gen", 100.0, lambda t: 1000 * t, realtime=False)
    outlet = generator._outlet

    generator.start()
    sleep(0.05)
    assert outlet.chunks == []

    generator.push_block(3)
    generator.push_block(2)
    generator.stop()

    assert outlet.chunks == [[[0], [10], [20]], [[30], [40]]]
    times = [ts for stamps in outlet.timestamps for ts in stamps]
    assert np.allclose(np.diff(times), 0.01)
    assert not generator.realtime


def test_generator_with_realtime_does_not_accept_requests():
    with pytest.raises(RuntimeError):
        LslGenerator("Gen", 100.0, lambda t: t).push_block(2)


def test_loopback_without_realtime_needs_a_generator_without_realtime(monkeypatch):
    monkeypatch.setattr(
        streams, "resolve_byprop", lambda *args, **kwargs: [FakeInfo("In", "data", 1, 500.0)]
    )

    with pytest.raises(ValueError):
        DeviceLoopback(lambda value: value, "In", realtime=False)
    with pytest.raises(ValueError):
        DeviceLoopback(
            lambda value: value, "In", realtime=False, generator=LslGenerator("Gen", 500.0, lambda t: t)
        )


class BlockGenerator:
    """Generator without real time which feeds the fake inlet of the loopback"""

    realtime = False

    def __init__(self, block_size: int) -> None:
        self.num_requests = 0
        self.block_size = block_size
        self.num_published = 0

    def push_block(self, num_samples: int) -> None:
        first = self.num_requests * num_samples
        FakeInlet.chunks.append(
            (
                [[first + idx] for idx in range(num_samples)],
                [100.0 + (first + idx) / 500.0 for idx in range(num_samples)],
            )
        )
        self.num_requests += 1


def test_loopback_without_realtime_requests_the_next_block_after_all_results(monkeypatch):
    monkeypatch.setattr(
        streams, "resolve_byprop", lambda *args, **kwargs: [FakeInfo("In", "data", 1, 500.0)]
    )
    generator = BlockGenerator(block_size=4)
    sent = []
    blocks = []

    def process(value: int):
        sent.append(value)
        if len(sent) % 4 == 0:
            blocks.append([[sent[-4 + idx] + 1] for idx in range(4)])

    def read_results():
        if blocks:
            assert generator.num_requests == len(sent) // 8 + (1 if len(sent) % 8 else 0)
            return blocks.pop(0)
        return []

    loopback = DeviceLoopback(
        process,
        "In",
        name_out="Out",
        read_results=read_results,
        realtime=False,
        generator=generator,
        block_samples=8,
    )

    loopback.start()
    _wait_until(lambda: len(loopback._outlet.chunks) >= 3)
    loopback.stop()

    assert loopback.exception is None
    assert loopback._outlet.chunks[0] == [[1], [2], [3], [4], [5], [6], [7], [8]]
    assert loopback._outlet.chunks[1] == [[9], [10], [11], [12], [13], [14], [15], [16]]
    assert loopback._outlet.timestamps[0] == [100.0 + idx / 500.0 for idx in range(8)]
    assert generator.num_requests >= 3


def test_loopback_without_realtime_uses_the_return_value_without_read_results(monkeypatch):
    monkeypatch.setattr(
        streams, "resolve_byprop", lambda *args, **kwargs: [FakeInfo("In", "data", 1, 500.0)]
    )
    generator = BlockGenerator(block_size=4)
    loopback = DeviceLoopback(
        lambda value: None if value == 1 else value * 2,
        "In",
        realtime=False,
        generator=generator,
        block_samples=4,
    )

    loopback.start()
    _wait_until(lambda: len(loopback._outlet.chunks) >= 2)
    loopback.stop()

    assert loopback._outlet.chunks[0] == [[0], [4], [6]]
    assert loopback._outlet.timestamps[0] == [100.0, 100.0 + 2 / 500.0, 100.0 + 3 / 500.0]
    assert loopback._num_no_response >= 1


def test_loopback_without_realtime_stops_with_an_error_if_the_device_does_not_respond(monkeypatch):
    monkeypatch.setattr(
        streams, "resolve_byprop", lambda *args, **kwargs: [FakeInfo("In", "data", 1, 500.0)]
    )
    loopback = DeviceLoopback(
        lambda value: None,
        "In",
        read_results=lambda: [],
        realtime=False,
        generator=BlockGenerator(block_size=4),
        block_samples=4,
        timeout_sec=0.05,
    )

    loopback.start()
    _wait_until(lambda: loopback.exception is not None)
    loopback.stop()

    assert isinstance(loopback.exception, TimeoutError)
    assert "block size" in str(loopback.exception)
    assert loopback._outlet.chunks == []


def test_error_monitor_reads_larger_chunks_without_realtime():
    from twin.streams import ErrorMonitor

    assert (
        ErrorMonitor("Device", "Twin")._pull_size
        < ErrorMonitor("Device", "Twin", realtime=False)._pull_size
    )
