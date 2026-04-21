import pytest
from api.helper import (
    _convert_pin_state,
    _convert_system_state,
    _convert_rp2_adc_value,
    _convert_rp2_temp_value,
    DataAcquisitionConfig
)


def test_pin_states():
    pin_list = ["LED_USER", "EN_PWR"]
    data_in = [idx for idx in range(2 ** len(pin_list))]
    check = ["NONE", "LED_USER", "EN_PWR", "LED_USER+EN_PWR"]
    rslt = [_convert_pin_state(idx, pin_list) for idx in data_in]
    assert rslt == check


def test_system_states():
    data_in = [idx for idx in range(6)]
    check = ["ERROR", "RESET", "INIT", "IDLE", "TEST", "DAQ"]
    rslt = [_convert_system_state(idx, check) for idx in data_in]
    assert rslt == check


def test_adc_rp2_value():
    data_in = [-1, 100, 2048, 4000, 4100]
    check = [0.0, 0.08058608058608059, 1.6504029304029304, 3.2234432234432235, 3.3]
    rslt = [_convert_rp2_adc_value(val) for val in data_in]
    assert rslt == check


def test_adc_rp2_temp():
    data_in = [-1, 100, 800, 850, 900, 2048, 4000, 4100]
    check = [437.22661243463097, 390.4014639244156, 62.625424352908325, 39.21285009780068, 15.800275842693033, -521.752429054579, -1435.7793279739822, -1480.2632190586867]
    rslt = [_convert_rp2_temp_value(val) for val in data_in]
    assert rslt == check


def test_daq_config_unsigned_2bytes():
    rslt = DataAcquisitionConfig(
        num_samples=2,
        num_channels=4,
        sampling_rate=2000.,
        send_batch=True,
        head_cmd=255,
        tail_cmd=160,
        num_bytes_total=1024,
        bytes_sample=2,
        is_signed=False
    )
    assert rslt.dtype_sample == '<u2'
    assert rslt.data_shape == (4, 2)


def test_daq_config_signed_2bytes():
    rslt = DataAcquisitionConfig(
        num_samples=600,
        num_channels=16,
        sampling_rate=2000.,
        send_batch=True,
        head_cmd=255,
        tail_cmd=160,
        num_bytes_total=1024,
        bytes_sample=2,
        is_signed=True
    )
    assert rslt.dtype_sample == '<i2'
    assert rslt.data_shape == (16, 600)


def test_daq_config_signed_4bytes():
    rslt = DataAcquisitionConfig(
        num_samples=600,
        num_channels=16,
        sampling_rate=2000.,
        send_batch=True,
        head_cmd=255,
        tail_cmd=160,
        num_bytes_total=1024,
        bytes_sample=4,
        is_signed=True
    )
    assert rslt.dtype_sample == '<i4'
    assert rslt.data_shape == (16, 600)


if __name__ == "__main__":
    pytest.main([__file__])
