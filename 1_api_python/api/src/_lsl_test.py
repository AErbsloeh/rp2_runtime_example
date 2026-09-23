from unittest.mock import patch

import pytest
import numpy as np

from pathlib import Path
from .plot_config_loader import load_plot_config
from shutil import rmtree
from time import sleep
from logging import basicConfig, DEBUG
import pylsl 
from api import get_path_to_project, DeviceAPI
from api.src._lsl import (
    RingBuffer,
    ThreadLSL
)
from ._live_visualizer import LivePlotter
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


# Fake classes for testing the plotting function without needing a real LSL stream or inlet
class FakeInfo:
    def channel_count(self):
        return 4
    
    def nominal_srate(self):#stream produces 250 sampkes per second
        return 250
    
class GeneratedDataInlet:
    def __init__(self):
        self.sample_index = 0 # index to keep track of the number of samples generated
        self._sample_rate = 250

    def info(self): # fake inlet need to have info function defined
        return FakeInfo()
    
    def pull_chunk(self, timeout=0.0, max_samples=1024):
        number_of_samples = min(5, max_samples)

        data = []
        timestamps = []

        for i in range(number_of_samples):
            current_sample_index = self.sample_index + i
            time=current_sample_index/self._sample_rate

            sample = [ 
                np.sin(2 * np.pi * 1 * time ),
                np.sin(2 * np.pi * 2 * time ),
                np.sin(2 * np.pi * 3 * time ),
                np.sin(2 * np.pi * 4 * time ),
             

               # current_sample_index,
                #current_sample_index + 10,
               
               # current_sample_index + 20,
                #current_sample_index + 30,
            ]

            timestamp = time

            data.append(sample)#adds 
            timestamps.append(timestamp)

        self.sample_index += number_of_samples

        return data, timestamps

def test_generated_plot():
    config_path = ( #creates a path to the live_plot_config.json file in the config directory of the project
        Path(__file__).resolve().parent.parent
        / "config"
        / "live_plot_config.json"
    )

    config = load_plot_config(config_path) #loads the plot configuration from the JSON file, returning a list of LivePlotterChannelConfig objects for enabled channels

    #for each channel in the configuration, a GeneratedDataInlet object is created to simulate an LSL inlet that generates data for that channel
    generated_inlets = [
        GeneratedDataInlet() for _ in config.channels
    
    ]
    #searches for the LSL stream and connects to it, returning the generated inlets for each channel
    with patch.object(
        LivePlotter,
        "_search_lsl_stream_and_connect",
        return_value=generated_inlets,
    ):
        #testing the real LivePlotter class with the generated data inlets, creating a live plot for each channel with the specified configuration
        plotter = LivePlotter(config=config)
        plotter.start()


# Real-hardware counterpart to test_generated_plot() above. Same config file,
# same LivePlotter, same plot - the only difference is that this one gets its
# data from an actual connected Pico instead of GeneratedDataInlet's fake sine
# waves. Nothing about test_generated_plot() or GeneratedDataInlet above was
# changed to add this - it's a separate, independent test.
#
# This will not find a stream yet: live_plot_config.json's channels still say
# lsl_layer_name = "GeneratedData", which no real hardware stream is named.
# It needs that value changed to "data" (the stream name start_daq() uses
# when do_process=False, same as we use in benchmark.py to avoid the
# __process_stream_func() 2-channel bug) before this test can connect.
def test_real_hardware_plot():
    config_path = (
        Path(__file__).resolve().parent.parent
        / "config"
        / "live_plot_config.json"
    )
    config = load_plot_config(config_path)

    DeviceAPI().do_reset()
    dut = DeviceAPI()
    dut.start_daq(
        sampling_rate=250,
        do_plot=False,
        do_process=False,  # skip the processing stage - __process_stream_func()
                            # is hardcoded for 2 channels, our DAQ has 8
        do_record=False,   # just streaming for the live plot, not saving to disk
    )

    try:
        # No patch.object() here, unlike test_generated_plot() above - this lets
        # LivePlotter's real _search_lsl_stream_and_connect() actually search for
        # and connect to the real LSL stream start_daq() just started.
        plotter = LivePlotter(config=config)
        plotter.start()
    finally:
        dut.stop_daq()
        dut.close()
class FakeLivePlotter:
    received_config = None # class variable to store the received configuration for plotting
    started = False # class variable to indicate whether the plotter has been started
    def __init__(self, config):# fake constructor to simulate the initialization of the plotter with a configuration
        FakeLivePlotter.received_config = config
    def start(self): # fake start method to simulate starting the plotter
        FakeLivePlotter.started = True

def test_plot_stream(monkeypatch):
    dut = ThreadLSL()
    dut._establish_lsl_inlet = lambda name: FakeInlet() #mock the _establish_lsl_inlet function to return a fake inlet with the necessary info for plotting
    monkeypatch.setattr('api.src._live_visualizer.LivePlotter', FakeLivePlotter) #mock the LivePlotter class to use the FakeLivePlotter instead of the real one
    dut.lsl_plot_stream(name='Playerdata', stim_idx=0)

    assert FakeLivePlotter.received_config is not None
    assert FakeLivePlotter.started == True
    assert len(FakeLivePlotter.received_config) == 4
    assert FakeLivePlotter.received_config[0].name == "C1"
    assert FakeLivePlotter.received_config[0].visualized_channel == 0
    assert FakeLivePlotter.received_config[2].lsl_layer_name == "Playerdata"

    
    
#if __name__ == "__main__":
 #   basicConfig(level=DEBUG)
  #  run_generated_plot()



