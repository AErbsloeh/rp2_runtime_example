from pathlib import Path
from shutil import rmtree

import numpy as np
import pytest

from api import DataAPI, DeviceAPI, StreamRecording, get_path_to_project


@pytest.fixture(scope="session", autouse=True)
def path():
    path = Path(get_path_to_project("temp_data"))
    rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)
    yield path
    rmtree(path, ignore_errors=True)


@pytest.fixture(scope="module", autouse=True)
def dut():
    DeviceAPI().do_reset()
    dut = DeviceAPI()
    dut.open()
    yield dut
    dut.do_reset()


@pytest.fixture(scope="session", autouse=True)
def period():
    period = 3.
    yield period


@pytest.fixture(autouse=True)
def stop_daq_after_test(dut: DeviceAPI):
    yield
    if dut.is_daq_running:
        dut.stop_daq()


@pytest.mark.hardware
def test_record_samples(path: Path, dut: DeviceAPI, period: float):
    dut.start_daq(
        folder_name="temp_data",
        sampling_rate=100.,
        window_sec=0.1,
        do_plot=False,
        do_record=True,
        do_process=True,
    )
    dut.wait_daq(period)
    dut.stop_daq()
    reader = DataAPI(path)
    reader.select_file(-1)

    assert 'filt' in reader.get_file_name()
    assert str(path) in reader.get_path2file()
    assert 'filt' in reader.get_overview_datatypes()

    data = reader.get_data('filt')
    assert type(data) == StreamRecording
    assert data.file == reader.get_path2file()
    assert data.num_channels == dut.get_daq_characteristics().num_channels
    assert data.sampling_rate == dut.get_daq_characteristics().sampling_rate
    assert data.data.shape[0] == dut.get_daq_characteristics().num_channels
    assert data.data.shape[1] == data.time.size
    assert 0.98 * data.get_sampling_rate < data.sampling_rate < 1.02 * data.sampling_rate
    assert data.time[-1] > period
    assert data.get_period_std < 5e-6

    assert reader.is_utilization_available
    util = reader.get_utilization()
    assert type(util) == StreamRecording
    assert util.file == reader.get_path2file()
    assert util.num_channels == 2
    assert 1.85 < util.sampling_rate < 2.15
    assert 1.85 < util.get_sampling_rate < 2.15
    assert util.data.shape[0] == 2
    assert util.data.shape[1] == util.time.size
    assert util.time[-1] > period
    assert util.get_period_std < 0.2


@pytest.mark.hardware
def test_record_batch(path: Path, dut: DeviceAPI, period: float):
    dut.start_daq(
        folder_name="temp_data",
        sampling_rate=100.,
        window_sec=0.1,
        do_plot=False,
        do_record=True,
        do_process=True,
    )
    dut.wait_daq(period)
    dut.stop_daq()
    reader = DataAPI(path)
    reader.select_file(-1)

    assert 'filt' in reader.get_file_name()
    assert str(path) in reader.get_path2file()
    assert 'filt' in reader.get_overview_datatypes()

    data = reader.get_data('filt')
    assert type(data) == StreamRecording
    assert data.file == reader.get_path2file()
    assert data.num_channels == dut.get_daq_characteristics().num_channels
    assert data.sampling_rate == dut.get_daq_characteristics().sampling_rate
    assert data.data.shape[0] == dut.get_daq_characteristics().num_channels
    assert data.data.shape[1] == data.time.size
    assert 0.98 * data.get_sampling_rate < data.sampling_rate < 1.02 * data.sampling_rate
    assert data.time[-1] > period
    assert data.get_period_std < 5e-6

    assert reader.is_utilization_available
    util = reader.get_utilization()
    assert type(util) == StreamRecording
    assert util.file == reader.get_path2file()
    assert util.num_channels == 2
    assert 1.9 < util.sampling_rate < 2.1
    assert 1.9 < util.get_sampling_rate < 2.1
    assert util.data.shape[0] == 2
    assert util.data.shape[1] == util.time.size
    assert util.time[-1] > period
    assert util.get_period_std < 0.2


@pytest.mark.hardware
def test_record_batch_with_layout(path: Path, dut: DeviceAPI, period: float):
    dut.start_daq(
        folder_name="temp_data",
        sampling_rate=100.,
        window_sec=0.1,
        do_plot=False,
        do_record=True,
        do_process=True,
    )
    dut.wait_daq(period)
    dut.stop_daq()
    reader = DataAPI(path)
    reader.select_file(-1)

    data = reader.get_data('filt')
    assert np.array_equal(data.label, np.asarray(['CH0', 'CH1']))


if __name__ == "__main__":
    pytest.main([__file__])
