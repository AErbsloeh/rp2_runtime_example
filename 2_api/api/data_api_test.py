from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from api import DataAcquisitionConfig, DataAPI, DeviceAPI, StreamRecording


@dataclass(frozen=True)
class Recording:
    folder: Path
    reader: DataAPI
    config_samples: DataAcquisitionConfig
    config_buffer: DataAcquisitionConfig
    period: float

    def get(self, name: str) -> StreamRecording:
        data = self.reader.get_data(name)
        assert type(data) == StreamRecording
        return data


@pytest.fixture(scope="module", autouse=True)
def dut():
    DeviceAPI().do_reset()
    dut = DeviceAPI()
    dut.open()
    yield dut
    dut.do_reset()


@pytest.fixture(scope="session", autouse=True)
def period():
    period = 3.0
    yield period


@pytest.fixture(autouse=True)
def stop_daq_after_test(dut: DeviceAPI):
    yield
    if dut.is_daq_running:
        dut.stop_daq()


@pytest.fixture(scope="module")
def recording(path: Path, dut: DeviceAPI, period: float) -> Recording:
    folder = path / "data_api"
    dut.start_daq(
        folder_name=str(folder),
        sampling_rate=100.0,
        window_sec=0.1,
        do_plot=False,
        do_record=True,
        do_process=True,
    )
    dut.wait_daq(period)
    dut.stop_daq()
    config_samples = dut.get_daq_characteristics(0)
    config_buffer = dut.get_daq_characteristics(1)

    reader = DataAPI(folder)
    reader.select_file(-1)
    return Recording(folder, reader, config_samples, config_buffer, period)


@pytest.mark.hardware
def test_recording_creates_one_file_with_all_streams(recording: Recording):
    files = recording.reader.get_overview_files

    assert len(files) == 1
    assert files[0].is_file()
    assert files[0].suffix == ".h5"
    assert files[0].stat().st_size > 0
    assert recording.folder in files[0].parents
    assert recording.reader.get_path2file() == str(files[0])
    assert "filt" in recording.reader.get_file_name()
    assert set(recording.reader.get_overview_datatypes()) == {"filt", "data0", "data1", "util"}


@pytest.mark.hardware
def test_sample_stream_matches_daq_config0(recording: Recording):
    config = recording.config_samples
    data = recording.get("data0")

    assert not config.send_batch
    assert data.file == recording.reader.get_path2file()
    assert data.num_channels == config.num_channels
    assert data.sampling_rate == config.sampling_rate
    assert data.data.shape == (config.num_channels, data.time.size)
    assert data.time.size > 1
    assert 0.98 * data.get_sampling_rate < data.sampling_rate < 1.02 * data.sampling_rate
    assert data.time[-1] > recording.period
    assert data.get_period_std < 5e-6


@pytest.mark.hardware
@pytest.mark.xfail(
    reason="the recording connects to its LSL streams later than the DAQ starts and loses the first samples",
    strict=False,
)
def test_recording_contains_the_samples_of_the_whole_measurement(recording: Recording):
    config = recording.config_samples
    data = recording.get("data0")

    assert data.time.size > 0.5 * recording.period * config.sampling_rate


@pytest.mark.hardware
def test_processed_stream_follows_the_sample_stream(recording: Recording):
    config = recording.config_samples
    samples = recording.get("data0")
    data = recording.get("filt")

    assert data.num_channels == config.num_channels
    assert data.sampling_rate == config.sampling_rate
    assert data.data.shape == (config.num_channels, data.time.size)
    assert abs(data.time.size - samples.time.size) <= 0.1 * samples.time.size
    assert 0.98 * data.get_sampling_rate < data.sampling_rate < 1.02 * data.sampling_rate


@pytest.mark.hardware
def test_buffer_stream_matches_daq_config1(recording: Recording):
    config = recording.config_buffer
    data = recording.get("data1")

    assert config.send_batch
    assert data.file == recording.reader.get_path2file()
    assert data.num_channels == config.num_channels
    assert data.sampling_rate == config.sampling_rate
    assert data.data.shape == (config.num_channels, data.time.size)
    assert data.time.size >= config.num_samples
    assert np.all(np.diff(data.time) >= 0)
    assert data.time[-1] - data.time[0] > 0.5 * recording.period


@pytest.mark.hardware
def test_utilization_is_recorded(recording: Recording):
    reader = recording.reader

    assert reader.is_utilization_available
    util = reader.get_utilization()
    assert type(util) == StreamRecording
    assert util.file == reader.get_path2file()
    assert util.num_channels == 2
    assert 1.85 < util.sampling_rate < 2.15
    assert 1.85 < util.get_sampling_rate < 2.15
    assert util.data.shape == (2, util.time.size)
    assert util.time[-1] > recording.period
    assert util.get_period_std < 0.2


@pytest.mark.hardware
@pytest.mark.parametrize("name", ["data0", "data1", "filt"])
def test_layout_information_is_transferred(recording: Recording, name: str):
    data = recording.get(name)

    assert list(map(str, data.label)) == [f"CH{idx}" for idx in range(data.num_channels)]
    assert list(map(str, data.units)) == ["LSB"] * data.num_channels
    assert len(data.layout) == data.num_channels
    assert all(str(entry) for entry in data.layout)


if __name__ == "__main__":
    pytest.main([__file__])
