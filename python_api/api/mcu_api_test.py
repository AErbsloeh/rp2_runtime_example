import pytest
from random import randint
from shutil import rmtree
from time import sleep
from api.mcu_api import (
    get_path_to_project,
    DeviceAPI
)


@pytest.fixture(scope="session", autouse=True)
def before_all_tests():
    DeviceAPI().do_reset()


@pytest.fixture(scope="session", autouse=True)
def after_all_tests():
    DeviceAPI().do_reset()
    rmtree(get_path_to_project("temp_data"), ignore_errors=True)


@pytest.fixture
def dut():
    mcu_api = DeviceAPI(
        com_name="AUTOCOM"
    )
    yield mcu_api


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


def test_check_system_state(dut: DeviceAPI):
    rslt = dut._get_system_state()
    assert rslt == "IDLE"


def test_check_system_clock(dut: DeviceAPI):
    rslt = dut._get_system_clock_khz()
    assert rslt in [125000, 150000]


def test_check_runtime(dut: DeviceAPI):
    wait_time_sec = 0.25

    time0 = dut._get_runtime_sec()
    sleep(wait_time_sec)
    time1 = dut._get_runtime_sec()
    assert time0 < time1
    assert 0.8 * wait_time_sec < time1 - time0 < 1.25 * wait_time_sec


def test_firmware_version(dut: DeviceAPI):
    rslt = dut._get_firmware_version()
    assert rslt == "0.1"


def test_temperature(dut: DeviceAPI):
    rslt = dut._get_temp_mcu()
    assert 20. < rslt < 36.


def test_check_system_state_class(dut: DeviceAPI):
    for _ in range(5):
        rslt = dut.get_state()
    assert rslt.system == "IDLE"
    assert rslt.pins == "LED_USER"
    assert rslt.runtime > 0
    assert rslt.clock in [125000, 150000]
    assert rslt.firmware == "0.1"
    assert 20. < rslt.temp < 36.


def test_enable_led(dut: DeviceAPI):
    dut.disable_led()
    assert dut._get_pin_state() == 'NONE'
    dut.enable_led()
    assert 'LED' in dut._get_pin_state()


def test_diable_led(dut: DeviceAPI):
    dut.enable_led()
    assert 'LED' in dut._get_pin_state()
    dut.disable_led()
    assert dut._get_pin_state() == 'NONE'
    

def test_toggle_led(dut: DeviceAPI):
    dut.disable_led()

    dut.toggle_led()
    assert 'LED' in dut._get_pin_state()
    dut.toggle_led()
    assert dut._get_pin_state() == 'NONE'
    
    dut.disable_led()


def test_control_daq_sample(dut: DeviceAPI):
    dut.update_daq_sampling_rate(1.)
    assert dut._get_system_state() == 'IDLE'

    dut.start_daq(do_batch=False, folder_name="temp_data")
    sleep(1.)
    dut.stop_daq()
    sleep(1.)


def test_control_daq_batch(dut: DeviceAPI):
    dut.update_daq_sampling_rate(1.)
    assert dut._get_system_state() == 'IDLE'

    dut.start_daq(do_batch=True, folder_name="temp_data")
    sleep(1.)
    dut.stop_daq()
    sleep(1.)


def test_channel_daq(dut: DeviceAPI):
    ret = dut._get_number_channels()
    assert ret == 2


def test_batchsize_daq(dut: DeviceAPI):
    ret = dut._get_number_samples_per_batch()
    assert ret == 16


def test_bytes_daq_sample(dut: DeviceAPI):
    ret = dut._get_number_bytes_per_daq_sample()
    assert ret == 15


def test_bytes_daq_batch(dut: DeviceAPI):
    ret = dut._get_number_bytes_per_daq_batch()
    assert ret == 83


if __name__ == "__main__":
    pytest.main([__file__])
