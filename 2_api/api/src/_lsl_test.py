import binascii
import os
import pty
import shutil
import tty
from logging import DEBUG, basicConfig
from pathlib import Path
from time import sleep

import numpy as np
import pylsl
import pytest
from pylsl import StreamInlet, resolve_bypred

from api.src import _c_reader
from api.src._c_reader import build_library
from api.src._helper import DaqStreamDescription, DataAcquisitionConfig
from api.src._lsl import RingBuffer, ThreadLSL, numpy_to_lsl_format

RUN_SECONDS = 3.0
CHECK_STEPS = 4
CHECK_SLEEP = 0.25


def test_lsl_format():
    dut = ThreadLSL()
    check = [
        pylsl.cf_float32,
        pylsl.cf_double64,
        pylsl.cf_string,
        pylsl.cf_int8,
        pylsl.cf_int16,
        pylsl.cf_int32,
        pylsl.cf_int64,
        pylsl.cf_undefined,
    ]
    results = ["float32", "float64", "string", "int8", "int16", "int32", "int64", ""]
    for input, format in zip(check, results):
        if input == pylsl.cf_undefined:
            try:
                dut._get_h5_format(input)
            except ValueError:
                assert True
            else:
                assert False
        else:
            assert dut._get_h5_format(input) == format


def test_ringbuffer_without_timestamp():
    dut = RingBuffer(5)
    buffer_in = np.zeros(shape=(5, 2))

    for idx in range(11):
        print(dut.get_data())
        np.testing.assert_array_equal(dut.get_data().shape, buffer_in.shape)
        dut.append(idx + 1)


def test_ringbuffer_with_timestamp():
    dut = RingBuffer(5)
    buffer_in = np.zeros(shape=(5, 2))

    for idx in range(11):
        print(dut.get_data())
        np.testing.assert_array_equal(dut.get_data().shape, buffer_in.shape)
        dut.append_with_timestamp(idx, idx + 1)


def test_thread_init():
    dut = ThreadLSL()
    assert not dut.is_alive
    assert not dut.is_running


def test_get_num_sampling_rates():
    dut = ThreadLSL()
    assert dut._get_number_stream_samples(10) == 10
    assert dut._get_number_stream_samples(499) == 10
    assert dut._get_number_stream_samples(501) == 10
    assert dut._get_number_stream_samples(1000) == 20


def test_thread_register_and_start():
    dut = ThreadLSL()
    assert len(dut._thread) == 0

    dut.register(func=dut._thread_dummy, args=(0,))
    assert len(dut._thread) == 2

    assert not dut.is_running
    dut.start()
    for ite in range(CHECK_STEPS):
        sleep(CHECK_SLEEP)
        assert dut.is_running
        dut._check_exception()
    dut.stop()
    assert not dut.is_running


def test_thread_register_and_abort():
    dut = ThreadLSL()
    assert len(dut._thread) == 0

    dut.register(func=dut._thread_dummy, args=(0,))
    assert len(dut._thread) == 2

    assert not dut.is_running
    try:
        dut.start()
        for ite in range(CHECK_STEPS):
            sleep(CHECK_SLEEP)
            dut._check_exception()
            print(ite, dut._is_active, dut._thread_active)
            if ite > 1:
                dut.stop()
                while dut._is_active:
                    sleep(0.1)
                raise RuntimeError
        assert not dut.is_running
    except RuntimeError:
        assert True
    else:
        assert not True


def test_thread_register_and_start_multiple():
    dut = ThreadLSL()
    assert len(dut._thread) == 0

    for idx in range(8):
        dut.register(func=dut._thread_dummy, args=(idx,))
    assert len(dut._thread) == 9

    assert not dut.is_running
    dut.start()
    for ite in range(CHECK_STEPS):
        sleep(CHECK_SLEEP)
        assert dut.is_running
        print(ite, dut._is_active, dut._thread_active)
        dut._check_exception()
    dut.stop()
    assert not dut.is_running


@pytest.mark.unstable
def test_thread_utilization(path: Path):
    dut = ThreadLSL()
    dut.register(func=dut.lsl_stream_util, args=(0, "util", 2.0))
    sleep(1)
    dut.register(func=dut.lsl_record_stream, args=(1, ["util"], path))
    assert len(dut._thread) == 3

    dut.start()
    dut.wait_for_seconds(RUN_SECONDS)
    dut.stop()
    assert not dut._is_active
    assert not dut.is_running


@pytest.mark.unstable
def test_thread_mock_random(path: Path):
    basicConfig(level=DEBUG)
    dut = ThreadLSL()
    channel_num = 4
    sample_rate = 4.0

    dut.register(func=dut.lsl_stream_util, args=(0, "util", 2.0))
    dut.register(func=dut.lsl_stream_mock, args=(1, "data", channel_num, sample_rate))
    dut.register(func=dut.lsl_record_stream, args=(2, ["data", "util"], path))
    assert len(dut._thread) == 4

    dut.start()
    dut.wait_for_seconds(RUN_SECONDS)
    dut.stop()
    assert not dut._is_active
    assert not dut.is_running


@pytest.mark.unstable
def test_thread_mock_file(path: Path):
    dut = ThreadLSL()
    dut.register(func=dut.lsl_stream_util, args=(0, "util", 1.0))
    dut.register(func=dut.lsl_stream_file, args=(1, "mock", path, "data", -1))
    dut.register(func=dut.lsl_record_stream, args=(2, ["mock", "util"], path))
    assert len(dut._thread) == 4

    dut.start()
    try:
        dut.wait_for_seconds(RUN_SECONDS)
    except RuntimeError:
        pass
    finally:
        dut.stop()
    assert not dut._is_active
    assert not dut.is_running


@pytest.mark.unstable
def test_thread_split_stream():
    dut = ThreadLSL()
    dut.register(func=dut.lsl_stream_util, args=(0, "util", 2.0))
    dut.register(func=dut.lsl_split_stream, args=(1, "util", ["out0", "out1"]))
    dut.register(func=dut.lsl_stream_check_equality, args=(2, ["util", "out0", "out1"], 4))
    assert len(dut._thread) == 4

    dut.start()
    dut.wait_for_seconds(RUN_SECONDS)
    dut.stop()
    assert not dut._is_active
    assert not dut.is_running


@pytest.mark.unstable
def test_thread_process_stream():
    def daq_init(srate: float) -> None:
        pass

    def daq_process(data: list) -> list:
        return data

    dut = ThreadLSL()
    dut.register(func=dut.lsl_stream_util, args=(0, "util", 2.0))
    dut.register(func=dut.lsl_split_stream, args=(1, "util", ["out0", "out1"]))
    dut.register(func=dut.lsl_process_stream, args=(2, "out0", "save", 4, daq_init, daq_process))
    dut.register(func=dut.lsl_stream_check_equality, args=(3, ["out1", "save"], 4))
    assert len(dut._thread) == 5

    dut.start()
    dut.wait_for_seconds(RUN_SECONDS)
    dut.stop()
    assert not dut._is_active
    assert not dut.is_running


def test_process_stream_without_channel_description(monkeypatch):
    """Streams of other LSL sources have no labels, units and types (pylsl returns None), the output needs defaults"""
    created = {}

    class FakeInfo:
        def nominal_srate(self):
            return 100.0

        def channel_count(self):
            return 2

        def type(self):
            return "data"

        def channel_format(self):
            return pylsl.cf_int32

        def get_channel_labels(self):
            return None

        def get_channel_units(self):
            return None

        def get_channel_types(self):
            return None

    class FakeInlet:
        def info(self):
            return FakeInfo()

        def pull_chunk(self, max_samples, timeout):
            return [], None

    dut = ThreadLSL()
    monkeypatch.setattr(dut, "_establish_lsl_inlet", lambda name: FakeInlet())

    def fake_outlet(**kwargs):
        created.update(kwargs)
        dut._event.clear()
        return object()

    monkeypatch.setattr(dut, "_establish_lsl_outlet", fake_outlet)
    dut._thread_active = [True]
    dut._event.set()
    dut.lsl_process_stream(0, "in", "out", 4, lambda srate: None, lambda data: data)

    assert created["channel_labels"] == ["CH0", "CH1"]
    assert created["channel_units"] == ["", ""]
    assert created["channel_layout"] == ["", ""]
    assert created["channel_num"] == 2


if __name__ == "__main__":
    pytest.main([__file__])


def test_plot_stream_keeps_plotter_alive_while_event_loop_is_running(monkeypatch):
    from api.src import _lsl

    state = {}

    class FakePlotter:
        def __init__(self, config, window_width_sec):
            state["config"] = config
            state["window"] = window_width_sec

    class FakeApp:
        def exec(self):
            state["alive"] = isinstance(dut._plotter, FakePlotter)

    dut = ThreadLSL()
    monkeypatch.setattr(_lsl, "LivePlotter", FakePlotter)
    monkeypatch.setattr(_lsl.QtWidgets.QApplication, "instance", staticmethod(lambda: FakeApp()))

    dut.lsl_plot_stream("config", window_length=4.0)

    assert state == {"config": "config", "window": 4.0, "alive": True}
    assert dut._plotter is None


@pytest.mark.parametrize(
    "dtype, expected",
    [
        (np.float32, pylsl.cf_float32),
        (np.float64, pylsl.cf_double64),
        (np.dtype("float16"), pylsl.cf_double64),
        (np.int8, pylsl.cf_int8),
        (np.int16, pylsl.cf_int16),
        (np.int32, pylsl.cf_int32),
        (np.int64, pylsl.cf_int64),
        (np.uint16, pylsl.cf_int64),
    ],
)
def test_numpy_to_lsl_format(dtype, expected):
    assert numpy_to_lsl_format(dtype) == expected


def test_numpy_to_lsl_format_rejects_unknown_types():
    with pytest.raises(ValueError):
        numpy_to_lsl_format(np.bool_)
    with pytest.raises(ValueError):
        numpy_to_lsl_format(np.dtype("U4"))


@pytest.mark.skipif(
    os.name != "posix" or not any(shutil.which(cc) for cc in ("cc", "gcc", "clang")),
    reason="C reader needs a POSIX system and a C compiler",
)
def test_stream_system_c_pushes_packets_of_two_streams_into_lsl(tmp_path, monkeypatch):
    path_lib = build_library(tmp_path / "libdaq_reader.so")
    monkeypatch.setattr(_c_reader, "_library", None)
    monkeypatch.setenv(_c_reader.ENV_LIBRARY_PATH, str(path_lib))
    config_sample = DataAcquisitionConfig(False, 100.0, 2, 1, 0xA0, 0xFF, 3 + 8 + 4 + 2, 2, True)
    config_batch = DataAcquisitionConfig(True, 100.0, 1, 4, 0xB0, 0xFF, 3 + 16 + 4 * 2 + 2, 2, False)
    suffix = str(os.getpid())
    streams = [
        DaqStreamDescription(
            f"c_sample_{suffix}", 0xA0, 100.0, ["A", "B"], ["LSB", "LSB"], [[0], [1]], False
        ),
        DaqStreamDescription(f"c_batch_{suffix}", 0xB0, 100.0, ["C"], ["LSB"], [[0]], True),
    ]

    def packet(config, index, timestamps, values):
        data = bytes([config.head_cmd, index])
        for timestamp in timestamps:
            data += timestamp.to_bytes(8, "little")
        for value in values:
            data += value.to_bytes(config.bytes_sample, "little", signed=config.is_signed)
        return data + binascii.crc_hqx(data, 0xFFFF).to_bytes(2, "little") + bytes([config.tail_cmd])

    master, slave = pty.openpty()
    tty.setraw(slave)
    finished = []
    dut = ThreadLSL()
    dut.register(
        dut.lsl_stream_system_c,
        args=(0, streams, [config_sample, config_batch], slave),
        kwargs={"on_finished": finished.append, "consumer_timeout": 5.0},
    )
    dut.start()
    try:
        inlets = []
        for stream in streams:
            info = resolve_bypred(f"name='{stream.name}'", timeout=5.0)
            assert len(info) == 1
            inlets.append(StreamInlet(info[0]))
            inlets[-1].open_stream(timeout=5.0)
        sleep(0.5)

        os.write(master, packet(config_sample, 0, [1_000_000], [-5, 7]))
        os.write(master, packet(config_batch, 0, [2_000_000, 2_030_000], [1, 2, 3, 65535]))
        os.write(master, packet(config_sample, 2, [1_010_000], [9, -9]))

        samples, stamps_sample = inlets[0].pull_chunk(timeout=3.0, max_samples=2)
        for _ in range(20):
            if len(samples) == 2:
                break
            more, more_stamps = inlets[0].pull_chunk(timeout=0.25, max_samples=2)
            samples += more
            stamps_sample += more_stamps
        chunk, stamps_batch = inlets[1].pull_chunk(timeout=3.0, max_samples=4)
        assert dut.is_running
    finally:
        dut.stop()
        os.close(master)
        os.close(slave)

    assert samples == [[-5, 7], [9, -9]]
    assert stamps_sample[1] - stamps_sample[0] == pytest.approx(0.01, abs=1e-3)
    assert chunk == [[1], [2], [3], [65535]]
    assert stamps_batch[-1] - stamps_batch[0] == pytest.approx(0.03, abs=1e-3)
    assert len(finished) == 1
    assert finished[0].num_packets == 3
    assert finished[0].num_lost_packets == 1
    assert not finished[0].running
