import pytest
import numpy as np
from pathlib import Path
from shutil import rmtree
from time import sleep
from logging import basicConfig, DEBUG
import pylsl

from api import get_path_to_project
from api.lsl import (
    RingBuffer,
    ThreadLSL
)

@pytest.fixture(scope='session', autouse=True)
def path():
    path = Path(get_path_to_project("temp_data"))
    rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)
    yield path
    rmtree(path, ignore_errors=True)


def test_lsl_format():
    dut = ThreadLSL()
    check = [pylsl.cf_float32, pylsl.cf_double64, pylsl.cf_string, pylsl.cf_int8, pylsl.cf_int16, pylsl.cf_int32, pylsl.cf_int64, pylsl.cf_undefined]
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

    for idx in range (11):
        print(dut.get_data())
        np.testing.assert_array_equal(dut.get_data().shape, buffer_in.shape)
        dut.append(idx+1)


def test_ringbuffer_with_timestamp():
    dut = RingBuffer(5)
    buffer_in = np.zeros(shape=(5, 2))

    for idx in range (11):
        print(dut.get_data())
        np.testing.assert_array_equal(dut.get_data().shape, buffer_in.shape)
        dut.append_with_timestamp(idx, idx+1)


def test_thread_init():
    dut = ThreadLSL()
    assert dut.is_alive == False
    assert dut.is_running == False


def test_get_num_sampling_rates():
    dut = ThreadLSL()
    assert dut._get_number_stream_samples(10) == 10
    assert dut._get_number_stream_samples(499) == 10
    assert dut._get_number_stream_samples(501) == 10
    assert dut._get_number_stream_samples(1000) == 20


def test_thread_register_and_start():
    dut = ThreadLSL()
    assert len(dut._thread) == 0

    dut.register(
        func=dut._thread_dummy,
        args=(0, )
    )
    assert len(dut._thread) == 2

    assert dut.is_running == False
    dut.start()
    for ite in range(10):
        sleep(0.5)
        assert dut.is_running == True
        dut._check_exception()
    dut.stop()
    assert dut.is_running == False


def test_thread_register_and_abort():
    dut = ThreadLSL()
    assert len(dut._thread) == 0

    dut.register(
        func=dut._thread_dummy,
        args=(0, )
    )
    assert len(dut._thread) == 2

    assert dut.is_running == False
    try:
        dut.start()
        for ite in range(10):
            sleep(0.5)
            dut._check_exception()
            print(ite, dut._is_active, dut._thread_active)
            if ite > 5:
                dut.stop()
                while dut._is_active:
                    sleep(0.1)
                raise RuntimeError
        assert dut.is_running == False
    except RuntimeError:
        assert True == True
    else:
        assert False == True


def test_thread_register_and_start_multiple():
    dut = ThreadLSL()
    assert len(dut._thread) == 0

    for idx in range(8):
        dut.register(func=dut._thread_dummy, args=(idx, ))
    assert len(dut._thread) == 9

    assert dut.is_running == False
    dut.start()
    for ite in range(10):
        sleep(0.5)
        assert dut.is_running == True
        print(ite, dut._is_active, dut._thread_active)
        dut._check_exception()
    dut.stop()
    assert dut.is_running == False


def test_thread_utilization(path: Path):
    dut = ThreadLSL()
    dut.register(func=dut.lsl_stream_util, args=(0, 'util', 2.))
    dut.register(func=dut.lsl_record_stream, args=(1, ['util'], path))
    assert len(dut._thread) == 3

    dut.start()
    dut.wait_for_seconds(10.)
    dut.stop()
    assert dut._is_active == False
    assert dut.is_running == False


def test_thread_mock_random(path: Path):
    dut = ThreadLSL()
    channel_num = 4
    sample_rate = 200

    dut.register(func=dut.lsl_stream_util, args=(0, 'util', 2.))
    dut.register(func=dut.lsl_stream_mock, args=(1, 'data', channel_num, sample_rate))
    dut.register(func=dut.lsl_record_stream, args=(2, ['data', 'util'], path))
    assert len(dut._thread) == 4

    dut.start()
    dut.wait_for_seconds(10.)
    dut.stop()
    assert dut._is_active == False
    assert dut.is_running == False


def test_thread_mock_file(path: Path):
    dut = ThreadLSL()
    dut.register(func=dut.lsl_stream_util, args=(0, 'util', 1.))
    dut.register(func=dut.lsl_stream_file, args=(1, 'mock', path, 'data', -1))
    dut.register(func=dut.lsl_record_stream, args=(2, ['mock', 'util'], path))
    assert len(dut._thread) == 4

    dut.start()
    try:
        dut.wait_for_seconds(10.)
    except RuntimeError:
        dut.stop()
        assert dut._is_active == False
        assert dut.is_running == False
    else:
        assert dut._is_active == True


def test_thread_split_stream():
    dut = ThreadLSL()
    dut.register(func=dut.lsl_stream_util, args=(0, 'util', 2.))
    dut.register(func=dut.lsl_split_stream, args=(1, 'util', ['out0', 'out1']))
    dut.register(func=dut.lsl_stream_check_equality, args=(2, ['util', 'out0', 'out1'], 4))
    assert len(dut._thread) == 4

    dut.start()
    dut.wait_for_seconds(10.)
    dut.stop()
    assert dut._is_active == False
    assert dut.is_running == False


def test_thread_process_stream():
    def daq_init(srate: float) -> None:
        pass

    def daq_process(data: list) -> list:
        return data

    dut = ThreadLSL()
    dut.register(func=dut.lsl_stream_util, args=(0, 'util', 2.))
    dut.register(func=dut.lsl_split_stream, args=(1, 'util', ['out0', 'out1']))
    dut.register(func=dut.lsl_process_stream, args=(2, 'out0', 'save', 4, daq_init, daq_process))
    dut.register(func=dut.lsl_stream_check_equality, args=(3, ['out1', 'save'], 4))
    assert len(dut._thread) == 5

    dut.start()
    dut.wait_for_seconds(10.)
    dut.stop()
    assert dut._is_active == False
    assert dut.is_running == False


if __name__ == "__main__":
    basicConfig(level=DEBUG)
    pytest.main([__file__])
