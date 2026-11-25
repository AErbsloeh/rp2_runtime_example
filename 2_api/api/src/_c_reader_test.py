import binascii
import os
import pty
import shutil
import threading
import time
import tty
from ctypes import CFUNCTYPE, POINTER, c_double, c_int32, c_ulong, c_void_p

import pytest

from api.src import _c_reader
from api.src._c_reader import CDaqReader, build_library, load_library
from api.src._helper import DataAcquisitionConfig

pytestmark = pytest.mark.skipif(
    os.name != "posix" or not any(shutil.which(cc) for cc in ("cc", "gcc", "clang")),
    reason="C reader needs a POSIX system and a C compiler",
)

PUSH_SAMPLE = CFUNCTYPE(c_int32, c_void_p, POINTER(c_int32), c_double, c_int32)
PUSH_CHUNK = CFUNCTYPE(c_int32, c_void_p, POINTER(c_int32), c_ulong, POINTER(c_double), c_int32)


def make_config(
    head, batch=False, signed=True, bytes_sample=2, crc=True, channels=3, samples=1, tail=0xFF
):
    num_bytes = 3 + (16 if batch else 8) + channels * samples * bytes_sample + (2 if crc else 0)
    return DataAcquisitionConfig(
        send_batch=batch,
        sampling_rate=1000.0,
        num_channels=channels,
        num_samples=samples,
        head_cmd=head,
        tail_cmd=tail,
        num_bytes_total=num_bytes,
        bytes_sample=bytes_sample,
        is_signed=signed,
    )


def make_packet(config, index, timestamps_us, values):
    packet = bytes([config.head_cmd, index])
    for timestamp in timestamps_us:
        packet += int(timestamp).to_bytes(8, "little")
    for value in values:
        packet += int(value).to_bytes(config.bytes_sample, "little", signed=config.is_signed)
    if config.has_crc:
        packet += binascii.crc_hqx(packet, 0xFFFF).to_bytes(2, "little")
    packet += bytes([config.tail_cmd])
    assert len(packet) == config.num_bytes_total
    return packet


class Sink:
    """Receiving the pushes of the C reader instead of LSL"""

    def __init__(self, num_values: dict[int, int], num_channels: dict[int, int] | None = None) -> None:
        self.num_values = num_values
        self.num_channels = num_channels if num_channels is not None else num_values
        self.samples = []
        self.chunks = []
        self.push_sample = PUSH_SAMPLE(self._push_sample)
        self.push_chunk = PUSH_CHUNK(self._push_chunk)

    def _push_sample(self, outlet, data, timestamp, pushthrough):
        self.samples.append((outlet, [data[idx] for idx in range(self.num_values[outlet])], timestamp))
        return 0

    def _push_chunk(self, outlet, data, num_elements, timestamps, pushthrough):
        num_samples = num_elements // self.num_channels[outlet]
        self.chunks.append(
            (
                outlet,
                [data[idx] for idx in range(num_elements)],
                [timestamps[idx] for idx in range(num_samples)],
            )
        )
        return 0


@pytest.fixture(scope="module")
def library(tmp_path_factory):
    path = build_library(tmp_path_factory.mktemp("c_reader") / "libdaq_reader.so")
    return load_library(path)


@pytest.fixture
def port():
    master, slave = pty.openpty()
    tty.setraw(slave)
    yield master, slave
    for fd in (master, slave):
        try:
            os.close(fd)
        except OSError:
            pass


def wait_for(condition, timeout=3.0):
    start = time.monotonic()
    while not condition():
        assert time.monotonic() - start < timeout, "timeout"
        time.sleep(0.005)


def make_reader(library, configs):
    outlets = list(range(1, len(configs) + 1))
    sink = Sink(
        {outlet: config.num_channels * config.num_samples for outlet, config in zip(outlets, configs)},
        {outlet: config.num_channels for outlet, config in zip(outlets, configs)},
    )
    return CDaqReader(configs, outlets, sink.push_sample, sink.push_chunk, library), sink


def test_sample_packets_are_decoded_with_sign_and_timestamp(library, port):
    master, slave = port
    config = make_config(0xA0, signed=True, bytes_sample=2, channels=3)
    reader, sink = make_reader(library, [config])
    reader.start(slave)
    for idx, values in enumerate([[1, -2, 3], [-32768, 32767, 0]]):
        os.write(master, make_packet(config, idx, [2_500_000 + idx], values))
    wait_for(lambda: len(sink.samples) == 2)
    reader.stop()

    assert sink.samples[0] == (1, [1, -2, 3], pytest.approx(2.5))
    assert sink.samples[1] == (1, [-32768, 32767, 0], pytest.approx(2.500001))
    stats = reader.stats()
    assert (stats.num_packets, stats.num_lost_packets, stats.num_skipped_bytes, stats.error) == (
        2,
        0,
        0,
        0,
    )


@pytest.mark.parametrize(
    ("bytes_sample", "signed", "value"),
    [(1, True, -100), (3, True, -8388608), (3, False, 16777215), (4, True, -5)],
)
def test_all_value_sizes_are_decoded(library, port, bytes_sample, signed, value):
    master, slave = port
    config = make_config(0xA1, signed=signed, bytes_sample=bytes_sample, channels=1, crc=False)
    reader, sink = make_reader(library, [config])
    reader.start(slave)
    os.write(master, make_packet(config, 0, [1], [value]))
    wait_for(lambda: sink.samples)
    reader.stop()

    assert sink.samples[0][1] == [value]


def test_batch_packet_is_pushed_as_chunk_with_interpolated_timestamps(library, port):
    master, slave = port
    config = make_config(0xB0, batch=True, signed=False, bytes_sample=2, channels=2, samples=4)
    reader, sink = make_reader(library, [config])
    reader.start(slave)
    values = [10, 11, 20, 21, 30, 31, 65535, 0]
    os.write(master, make_packet(config, 7, [1_000_000, 1_030_000], values))
    wait_for(lambda: sink.chunks)
    reader.stop()

    outlet, data, timestamps = sink.chunks[0]
    assert outlet == 1
    assert data == values
    assert timestamps == pytest.approx([1.0, 1.01, 1.02, 1.03])


def test_streams_in_any_order_are_sent_to_their_outlets(library, port):
    master, slave = port
    config_a = make_config(0xA0, channels=2)
    config_b = make_config(0xB0, batch=True, crc=False, channels=1, samples=2, bytes_sample=4)
    reader, sink = make_reader(library, [config_a, config_b])
    reader.start(slave)
    os.write(
        master,
        make_packet(config_a, 0, [1], [1, 2])
        + make_packet(config_b, 0, [10, 20], [7, 8])
        + make_packet(config_b, 1, [30, 40], [9, 10])
        + make_packet(config_a, 1, [2], [3, 4]),
    )
    wait_for(lambda: len(sink.samples) == 2 and len(sink.chunks) == 2)
    reader.stop()

    assert [(outlet, values) for outlet, values, _ in sink.samples] == [(1, [1, 2]), (1, [3, 4])]
    assert [(outlet, values) for outlet, values, _ in sink.chunks] == [(2, [7, 8]), (2, [9, 10])]


def test_reader_resynchronises_after_garbage_bad_crc_and_fragmented_writes(library, port):
    master, slave = port
    config = make_config(0xA0, channels=2)
    reader, sink = make_reader(library, [config])
    reader.start(slave)
    bad = bytearray(make_packet(config, 1, [2], [9, 9]))
    bad[5] ^= 0xFF
    good = make_packet(config, 2, [3], [5, 6])
    os.write(master, b"\x00\x13\xa0" + make_packet(config, 0, [1], [1, 2]) + bytes(bad))
    for piece in (good[:4], good[4:9], good[9:]):
        os.write(master, piece)
        time.sleep(0.02)
    wait_for(lambda: len(sink.samples) == 2)
    reader.stop()

    assert [values for _, values, _ in sink.samples] == [[1, 2], [5, 6]]
    stats = reader.stats()
    assert stats.num_packets == 2
    assert stats.num_skipped_bytes >= 3 + len(bad)


def test_lost_packets_are_counted_including_wrap_of_the_index(library, port):
    master, slave = port
    config = make_config(0xA0, channels=1, crc=False)
    reader, sink = make_reader(library, [config])
    reader.start(slave)
    for index in (0, 1, 4, 254, 255, 0, 3):
        os.write(master, make_packet(config, index, [index], [index]))
    wait_for(lambda: len(sink.samples) == 7)
    reader.stop()

    assert reader.stats().num_lost_packets == 2 + 249 + 2


def test_reader_reports_the_error_when_the_connection_is_closed(library, port):
    master, slave = port
    config = make_config(0xA0, channels=1)
    reader, _ = make_reader(library, [config])
    reader.start(slave)
    os.close(master)
    wait_for(lambda: not reader.stats().running)
    reader.stop()

    assert reader.stats().error != 0


def test_reader_stops_fast_without_data(library, port):
    _, slave = port
    reader, _ = make_reader(library, [make_config(0xA0)])
    reader.start(slave)
    time.sleep(0.1)
    start = time.monotonic()
    reader.stop()
    assert time.monotonic() - start < 0.5
    assert not reader.stats().running


def test_invalid_configurations_are_rejected(library):
    sink = Sink({1: 3, 2: 3})
    config = make_config(0xA0)
    with pytest.raises(ValueError):
        CDaqReader([config, config], [1, 2], sink.push_sample, sink.push_chunk, library)
    wrong_size = make_config(0xA0)
    object.__setattr__(wrong_size, "num_bytes_total", wrong_size.num_bytes_total + 1)
    with pytest.raises(ValueError):
        CDaqReader([wrong_size], [1], sink.push_sample, sink.push_chunk, library)
    with pytest.raises(ValueError):
        CDaqReader([config], [], sink.push_sample, sink.push_chunk, library)


def test_library_is_found_after_building(tmp_path, monkeypatch):
    path = build_library(tmp_path / "libdaq_reader.so")
    monkeypatch.setenv(_c_reader.ENV_LIBRARY_PATH, str(path))
    assert _c_reader.find_library() == path


def test_reader_does_not_block_other_python_threads(library, port):
    master, slave = port
    config = make_config(0xA0, channels=1, crc=False)
    reader, sink = make_reader(library, [config])
    reader.start(slave)
    counter = []
    thread = threading.Thread(target=lambda: [counter.append(1) for _ in range(1000)])
    thread.start()
    for index in range(200):
        os.write(master, make_packet(config, index % 256, [index], [index]))
    thread.join()
    wait_for(lambda: len(sink.samples) == 200)
    reader.stop()
    assert len(counter) == 1000


def test_build_uses_fallback_folder_if_build_folder_is_not_usable(tmp_path, monkeypatch):
    blocker = tmp_path / "file"
    blocker.write_text("")
    fallback = tmp_path / "fallback"
    monkeypatch.setattr(_c_reader, "_build_folders", lambda: [blocker / "build", fallback])
    monkeypatch.delenv(_c_reader.ENV_LIBRARY_PATH, raising=False)

    path = build_library()

    assert path.parent == fallback
    assert _c_reader.find_library() == path
