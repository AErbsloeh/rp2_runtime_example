import pytest
from random import randint
from shutil import rmtree
from time import sleep
from logging import basicConfig, DEBUG
from api.mcu_api import (
    get_path_to_project,
    DeviceAPI
)
from api.helper import (
    _convert_pin_state,
    _convert_system_state
)


@pytest.fixture(scope="session", autouse=True)
def dut():
    DeviceAPI().do_reset()
    mcu_api = DeviceAPI(
        com_name="AUTOCOM"
    )
    mcu_api.open()
    yield mcu_api
    mcu_api.do_reset()
    path = get_path_to_project("temp_data")
    rmtree(path, ignore_errors=True)


def test_num_bytes(dut: DeviceAPI):
    assert dut.total_num_bytes == 3


def test_check_opened(dut: DeviceAPI):
    assert dut.is_com_port_active == True


def test_check_echo(dut: DeviceAPI):
    test_pattern = "TESTS"
    ret = dut.echo(test_pattern)
    assert ret == test_pattern
    assert len(ret) == len(test_pattern)


def test_bytes_int_conversion(dut: DeviceAPI):
    input = [randint(a=0, b=65535) for _ in range(100)]
    rslt = []
    for val in input:
        data = val.to_bytes(2, byteorder='little')
        rslt.append(dut._bytes_to_int(data))
    assert rslt == input


def test_charac_system(dut: DeviceAPI):
    rslt = dut.get_state()
    assert rslt.system == "IDLE"
    assert rslt.clock in [125000, 150000]
    assert rslt.firmware == "0.1"
    assert 20. < rslt.temp < 36.


def test_check_runtime(dut: DeviceAPI):
    wait_time_sec = 0.25

    time0 = dut.get_state().runtime
    sleep(wait_time_sec)
    time1 = dut.get_state().runtime
    assert time0 < time1
    assert 0.8 * wait_time_sec < time1 - time0 < 1.25 * wait_time_sec


def test_enable_led(dut: DeviceAPI):
    dut.disable_led()
    assert dut.get_state().pins == 'NONE'
    dut.enable_led()
    assert 'LED' in dut.get_state().pins


def test_diable_led(dut: DeviceAPI):
    dut.enable_led()
    assert 'LED' in dut.get_state().pins
    dut.disable_led()
    assert dut.get_state().pins == 'NONE'
    

def test_toggle_led(dut: DeviceAPI):
    dut.disable_led()

    dut.toggle_led()
    assert 'LED' in dut.get_state().pins
    dut.toggle_led()
    assert dut.get_state().pins == 'NONE'
    
    dut.disable_led()


def test_control_daq_sample(dut: DeviceAPI):
    assert dut.get_state().system == 'IDLE'

    dut.start_daq(sampling_rate=1., do_batch=False, folder_name="temp_data")
    sleep(1.)
    dut.stop_daq()


def test_control_daq_batch(dut: DeviceAPI):
    assert dut.get_state().system == 'IDLE'

    dut.start_daq(sampling_rate=1., do_batch=True, folder_name="temp_data")
    sleep(1.)
    assert dut.is_daq_running == True
    dut.stop_daq()


def test_config_daq_sample_100hz(dut: DeviceAPI):
    dut._enable_batch_daq(False)
    dut._update_daq_sampling_rate(100.)

    rslt = dut.get_daq_characteristics()
    assert rslt.num_channels == 2
    assert rslt.num_samples == 1
    assert rslt.bytes_sample == 2
    assert rslt.sampling_rate == 100.
    assert rslt.send_batch == False
    assert rslt.num_bytes_total == 15
    assert rslt.tail_cmd == 255
    assert rslt.head_cmd == 160
    assert rslt.is_signed == False
    assert rslt.data_shape == (2,)
    assert rslt.dtype_sample == '<u2'


def test_config_daq_batch_10hz(dut: DeviceAPI):
    dut._enable_batch_daq(True)
    dut._update_daq_sampling_rate(10.)

    rslt = dut.get_daq_characteristics()
    assert rslt.num_channels == 2
    assert rslt.num_samples == 16
    assert rslt.bytes_sample == 2
    assert rslt.sampling_rate == 10.
    assert rslt.send_batch == True
    assert rslt.num_bytes_total == 83
    assert rslt.tail_cmd == 255
    assert rslt.head_cmd == 160
    assert rslt.data_shape == (2, 16)
    assert rslt.dtype_sample == '<u2'


def test_pin_states(dut: DeviceAPI):
    data_in = [idx for idx in range(2)]
    check = ["NONE", "LED_USER"]
    rslt = [_convert_pin_state(idx, dut._pin_names) for idx in data_in]
    assert rslt == check


def test_system_states(dut: DeviceAPI):
    data_in = [idx for idx in range(6)]
    check = ["ERROR", "RESET", "INIT", "IDLE", "TEST", "DAQ"]
    rslt = [_convert_system_state(idx, dut._state_names) for idx in data_in]
    assert rslt == check


def test_define_layout(dut: DeviceAPI):
    sets = dut.get_daq_characteristics()
    result_layout = [idx for idx in range(sets.num_channels)]
    result_labels = [f"CH{idx}" for idx in range(sets.num_channels)]

    dut.define_channel_layout(
        channel_layout=result_layout,
        channel_names=result_labels
    )
    check_layout, check_labels = dut.get_channel_layout()
    assert check_layout == result_layout
    assert check_labels == result_labels


if __name__ == "__main__":
    basicConfig(level=DEBUG)
    pytest.main([__file__])
