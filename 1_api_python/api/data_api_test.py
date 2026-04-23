import pytest
from pathlib import Path
from shutil import rmtree
from api import get_path_to_project, DataAPI, RawRecording, DeviceAPI


@pytest.fixture(scope="session", autouse=True)
def path():
    path = Path(get_path_to_project("temp_data"))
    rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)
    yield path
    rmtree(path, ignore_errors=True)


@pytest.fixture(scope="session", autouse=True)
def dut():
    DeviceAPI().do_reset()
    dut = DeviceAPI()
    yield dut
    dut.do_reset()


def test_record_samples_with_util(path: Path, dut: DeviceAPI):
    period_recording = 5.
    dut.define_channel_layout(
        channel_layout=[],
        channel_names=[]
    )
    dut.start_daq(
        folder_name="temp_data",
        sampling_rate=100.,
        window_sec=0.1,
        do_plot=False,
        do_batch=False,
        track_util=True
    )
    dut.wait_daq(period_recording)
    dut.stop_daq()
    reader = DataAPI(path)
    reader.select_file(-1)

    assert 'data' in reader.get_file_name()
    assert str(path) in reader.get_path2file()
    assert reader.is_layout_available == False

    data = reader.get_sensor_data()
    assert type(data) == RawRecording
    assert data.file == reader.get_path2file()
    assert data.num_channels == dut.get_daq_characteristics().num_channels
    assert data.sampling_rate == dut.get_daq_characteristics().sampling_rate
    assert data.data.shape[0] == dut.get_daq_characteristics().num_channels
    assert data.data.shape[1] == data.time.size
    assert 0.98 * data.get_sampling_rate < data.sampling_rate < 1.02 * data.sampling_rate
    assert data.time[-1] > period_recording
    assert data.get_period_std < 5e-6

    assert reader.is_utilization_available == True
    util = reader.get_utilization()
    assert type(util) == RawRecording
    assert util.file == reader.get_path2file()
    assert util.num_channels == 2
    assert 1.85 < util.sampling_rate < 2.15
    assert 1.85 < util.get_sampling_rate < 2.15
    assert util.data.shape[0] == 2
    assert util.data.shape[1] == util.time.size
    assert util.time[-1] > period_recording
    assert util.get_period_std < 0.2


def test_record_batch_with_util(path: Path, dut: DeviceAPI):
    period_recording = 5.
    dut.define_channel_layout(
        channel_layout=[],
        channel_names=[]
    )
    dut.start_daq(
        folder_name="temp_data",
        sampling_rate=100.,
        window_sec=0.1,
        do_plot=False,
        do_batch=True,
        track_util=True
    )
    dut.wait_daq(period_recording)
    dut.stop_daq()
    reader = DataAPI(path)
    reader.select_file(-1)

    assert 'data' in reader.get_file_name()
    assert str(path) in reader.get_path2file()
    assert reader.is_layout_available == False

    data = reader.get_sensor_data()
    assert type(data) == RawRecording
    assert data.file == reader.get_path2file()
    assert data.num_channels == dut.get_daq_characteristics().num_channels
    assert data.sampling_rate == dut.get_daq_characteristics().sampling_rate
    assert data.data.shape[0] == dut.get_daq_characteristics().num_channels
    assert data.data.shape[1] == data.time.size
    assert 0.98 * data.get_sampling_rate < data.sampling_rate < 1.02 * data.sampling_rate
    assert data.time[-1] > period_recording
    assert data.get_period_std < 5e-6

    assert reader.is_utilization_available == True
    util = reader.get_utilization()
    assert type(util) == RawRecording
    assert util.file == reader.get_path2file()
    assert util.num_channels == 2
    assert 1.9 < util.sampling_rate < 2.1
    assert 1.9 < util.get_sampling_rate < 2.1
    assert util.data.shape[0] == 2
    assert util.data.shape[1] == util.time.size
    assert util.time[-1] > period_recording
    assert util.get_period_std < 0.2


def test_record_samples_without_util(path: Path, dut: DeviceAPI):
    period_recording = 5.
    dut.define_channel_layout(
        channel_layout=[],
        channel_names=[]
    )
    dut.start_daq(
        folder_name="temp_data",
        sampling_rate=100.,
        window_sec=0.1,
        do_plot=False,
        do_batch=False,
        track_util=False
    )
    dut.wait_daq(period_recording)
    dut.stop_daq()
    reader = DataAPI(path)
    reader.select_file(-1)

    assert 'data' in reader.get_file_name()
    assert str(path) in reader.get_path2file()
    assert reader.is_layout_available == False

    data = reader.get_sensor_data()
    assert type(data) == RawRecording
    assert data.file == reader.get_path2file()
    assert data.num_channels == dut.get_daq_characteristics().num_channels
    assert data.sampling_rate == dut.get_daq_characteristics().sampling_rate
    assert data.data.shape[0] == dut.get_daq_characteristics().num_channels
    assert data.data.shape[1] == data.time.size
    assert 0.98 * data.get_sampling_rate < data.sampling_rate < 1.02 * data.sampling_rate
    assert data.time[-1] > period_recording

    assert reader.is_utilization_available == False
    try:
        reader.get_utilization()
    except Exception:
        assert True
    else:
        assert False


def test_record_batch_without_util(path: Path, dut: DeviceAPI):
    period_recording = 5.
    dut.define_channel_layout(
        channel_layout=[],
        channel_names=[]
    )
    dut.start_daq(
        folder_name="temp_data",
        sampling_rate=100.,
        window_sec=0.1,
        do_plot=False,
        do_batch=True,
        track_util=False
    )
    dut.wait_daq(period_recording)
    dut.stop_daq()
    reader = DataAPI(path)
    reader.select_file(-1)

    assert 'data' in reader.get_file_name()
    assert str(path) in reader.get_path2file()
    assert reader.is_layout_available == False

    data = reader.get_sensor_data()
    assert type(data) == RawRecording
    assert data.file == reader.get_path2file()
    assert data.num_channels == dut.get_daq_characteristics().num_channels
    assert data.sampling_rate == dut.get_daq_characteristics().sampling_rate
    assert data.data.shape[0] == dut.get_daq_characteristics().num_channels
    assert data.data.shape[1] == data.time.size
    assert 0.98 * data.get_sampling_rate < data.sampling_rate < 1.02 * data.sampling_rate
    assert data.time[-1] > period_recording
    assert data.get_period_std < 5e-6

    assert reader.is_utilization_available == False
    try:
        reader.get_utilization()
    except Exception:
        assert True
    else:
        assert False

def test_record_batch_with_layout(path: Path, dut: DeviceAPI):
    period_recording = 5.
    dut.define_channel_layout(
        channel_layout=[0, 1],
        channel_names=['0', '1']
    )
    dut.start_daq(
        folder_name="temp_data",
        sampling_rate=100.,
        window_sec=0.1,
        do_plot=False,
        do_batch=True,
        track_util=False
    )
    dut.wait_daq(period_recording)
    dut.stop_daq()
    reader = DataAPI(path)
    reader.select_file(-1)

    assert reader.is_layout_available == True
    assert reader.get_layout_information() == (['0', '1'], [0, 1])


if __name__ == "__main__":
    pytest.main([__file__])
