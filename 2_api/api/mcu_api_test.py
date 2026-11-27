from logging import DEBUG, basicConfig
from pathlib import Path
from random import randint
from time import sleep

import numpy as np
import pytest

from api.definitions import DaqDataType
from api.mcu_api import DeviceAPI
from api.src._helper import build_crc16_ccitt, convert_pin_state, convert_system_state


@pytest.fixture(scope="session", autouse=True)
def dut():
    DeviceAPI().do_reset()
    mcu_api = DeviceAPI(com_name="AUTOCOM")
    mcu_api.open()
    yield mcu_api


@pytest.mark.hardware
def test_num_bytes(dut: DeviceAPI):
    assert dut.total_num_bytes == dut._num_bytes_head + dut._num_bytes_data


@pytest.mark.hardware
def test_check_opened(dut: DeviceAPI):
    assert dut.is_com_port_active


@pytest.mark.hardware
def test_check_echo(dut: DeviceAPI):
    test_pattern = "TESTS"
    ret = dut.echo(test_pattern)
    assert ret == test_pattern
    assert len(ret) == len(test_pattern)


@pytest.mark.hardware
def test_check_echo_value(dut: DeviceAPI):
    for value in (0, 1, -1, 32768, -8388608, 8388607):
        assert dut.echo_value(value) == value


@pytest.mark.hardware
def test_check_process_data_template(dut: DeviceAPI):
    for value in (0, 1, -1, 32768, -8388608, 8388607):
        assert dut.process_data(value) is None


@pytest.mark.hardware
def test_get_processing_layout(dut: DeviceAPI):
    layout = dut.get_processing_layout()
    assert (layout.in_channels, layout.in_samples, layout.in_bytes_value) == (2, 1, 2)
    assert (layout.out_channels, layout.out_samples, layout.out_bytes_value) == (1, 1, 2)


@pytest.mark.hardware
def test_apply_data_array_and_process(dut: DeviceAPI):
    layout = dut.get_processing_layout()
    block = np.arange(layout.in_channels * layout.in_samples, dtype="<u2").reshape(
        layout.in_channels, layout.in_samples
    )
    dut.apply_data(block)
    assert dut.process_data() is None
    assert dut.process_data(block) is None


@pytest.mark.hardware
def test_bytes_int_conversion(dut: DeviceAPI):
    input = [randint(a=0, b=65535) for _ in range(100)]
    rslt = []
    for val in input:
        data = val.to_bytes(2, byteorder="little")
        rslt.append(dut._bytes_to_int(data))
    assert rslt == input


@pytest.mark.hardware
def test_charac_system(dut: DeviceAPI):
    rslt = dut.get_state()
    assert rslt.system == "IDLE"
    assert rslt.clock in [125000, 150000]
    assert rslt.firmware == "0.1"
    assert 20.0 < rslt.temp < 36.0


@pytest.mark.hardware
def test_check_runtime(dut: DeviceAPI):
    wait_time_sec = 0.25

    time0 = dut.get_state().runtime
    sleep(wait_time_sec)
    time1 = dut.get_state().runtime
    assert time0 < time1
    assert 0.8 * wait_time_sec < time1 - time0 < 1.25 * wait_time_sec


@pytest.mark.hardware
def test_enable_led(dut: DeviceAPI):
    dut.disable_led()
    assert dut.get_state().pins == "NONE"
    dut.enable_led()
    assert "LED" in dut.get_state().pins


@pytest.mark.hardware
def test_diable_led(dut: DeviceAPI):
    dut.enable_led()
    assert "LED" in dut.get_state().pins
    dut.disable_led()
    assert dut.get_state().pins == "NONE"


@pytest.mark.hardware
def test_toggle_led(dut: DeviceAPI):
    dut.disable_led()

    dut.toggle_led()
    assert "LED" in dut.get_state().pins
    dut.toggle_led()
    assert dut.get_state().pins == "NONE"

    dut.disable_led()


@pytest.mark.hardware
@pytest.mark.order(-1)
def test_run_daq(dut: DeviceAPI, path: Path):
    DeviceAPI().do_reset()
    dut = DeviceAPI()
    assert dut.get_state().system == "IDLE"

    dut.start_daq(sampling_rate=250.0, folder_name=str(path), do_plot=False)
    sleep(1.0)
    assert dut.is_daq_running
    dut.stop_daq()


@pytest.mark.hardware
def test_get_number_daq_tasks(dut: DeviceAPI):
    num = dut._get_number_daq_tasks()
    assert num == 2


@pytest.mark.hardware
def test_config_daq_sample_100hz(dut: DeviceAPI):
    dut._update_daq_sampling_rate(100.0)

    rslt = dut.get_daq_characteristics(0)
    assert rslt.num_channels == 2
    assert rslt.num_samples == 1
    assert rslt.bytes_sample == 2
    assert rslt.sampling_rate == 100.0
    assert not rslt.send_batch
    assert rslt.num_bytes_total == 17
    assert rslt.tail_cmd == 255
    assert rslt.head_cmd == 160
    assert not rslt.is_signed
    assert rslt.data_shape == (2,)
    assert rslt.dtype_sample == "<u2"
    assert not rslt.is_signed
    assert rslt.has_crc
    assert rslt.data_type == DaqDataType.SAMPLE


@pytest.mark.hardware
def test_config_daq_batch_10hz(dut: DeviceAPI):
    dut._update_daq_sampling_rate(10.0)

    rslt = dut.get_daq_characteristics(1)
    assert rslt.num_channels == 1
    assert rslt.num_samples == 64
    assert rslt.bytes_sample == 2
    assert rslt.sampling_rate == 10.0
    assert rslt.send_batch
    assert rslt.num_bytes_total == 149
    assert rslt.tail_cmd == 0xBC
    assert rslt.head_cmd == 0xFA
    assert rslt.data_shape == (1, 64)
    assert rslt.dtype_sample == "<u2"
    assert not rslt.is_signed
    assert rslt.has_crc
    assert rslt.data_type == DaqDataType.SAMPLE


@pytest.mark.hardware
def test_pin_states(dut: DeviceAPI):
    data_in = [idx for idx in range(2)]
    check = ["NONE", "LED_USER"]
    rslt = [convert_pin_state(idx, dut._pin_names) for idx in data_in]
    assert rslt == check


@pytest.mark.hardware
def test_system_states(dut: DeviceAPI):
    data_in = [idx for idx in range(6)]
    check = ["ERROR", "RESET", "INIT", "IDLE", "TEST", "DAQ"]
    rslt = [convert_system_state(idx, dut._state_names) for idx in data_in]
    assert rslt == check


@pytest.mark.hardware
def test_check_crc_valid(dut: DeviceAPI):
    payload = b"123456789"
    crc = build_crc16_ccitt(payload)
    packet = payload + crc.to_bytes(2, byteorder="little") + bytes([0xA0])
    assert dut._check_crc(packet, crc)


@pytest.mark.hardware
def test_check_crc_invalid(dut: DeviceAPI):
    payload = b"123456789"
    crc = build_crc16_ccitt(payload)
    wrong_crc = crc + 1
    packet = payload + wrong_crc.to_bytes(2, byteorder="little") + bytes([0xA0])
    try:
        dut._check_crc(packet, wrong_crc)
    except RuntimeError:
        assert True
    else:
        assert False


if __name__ == "__main__":
    basicConfig(level=DEBUG)
    pytest.main([__file__])
