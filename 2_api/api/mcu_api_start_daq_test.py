from logging import getLogger

import numpy as np
import pytest

from api.mcu_api import (
    Commands,
    DeviceAPI,
)
from api.src._c_reader import DaqReaderStats
from api.src._helper import DataAcquisitionConfig, ProcessingLayout


class FakeDevice:
    timeout: float

    def __init__(self) -> None:
        self.timeout = 0.0

    def fileno(self) -> int:
        return 42


class FakeThreads:
    def __init__(self) -> None:
        self.registered = []
        self.started = False
        self.plotted = None

    def register(self, func, args, kwargs=None) -> None:
        self.registered.append((func.__name__, args, kwargs or {}))

    def start(self) -> None:
        self.started = True

    def lsl_stream_util(self) -> None:
        pass

    def lsl_stream_system(self) -> None:
        pass

    def lsl_stream_system_c(self) -> None:
        pass

    def lsl_process_stream(self) -> None:
        pass

    def lsl_record_stream(self) -> None:
        pass

    def lsl_plot_stream(self, plot_config=None, window_length=0.0) -> None:
        self.plotted = (plot_config, window_length)


def _make_daq_config() -> DataAcquisitionConfig:
    return DataAcquisitionConfig(
        send_batch=True,
        sampling_rate=500.0,
        num_channels=2,
        num_samples=16,
        head_cmd=0xA0,
        tail_cmd=0xFF,
        num_bytes_total=85,
        bytes_sample=2,
        is_signed=False,
    )


def _make_device_api():
    dut = DeviceAPI.__new__(DeviceAPI)
    threads = FakeThreads()
    device = FakeDevice()
    calls = []

    dut._threads = threads
    dut._device = device
    dut._logger = getLogger(__name__)
    dut._plot_config = None
    dut._num_package_loss = 0

    dut._update_daq_sampling_rate = lambda sampling_rate: calls.append(("rate", sampling_rate))
    dut._get_number_daq_tasks = lambda: 1
    dut.get_daq_characteristics = lambda sel_daq_tasks=0: _make_daq_config()
    dut._write_without_feedback = lambda head, data=0: calls.append(("write", head, data))

    return dut, threads, device, calls


@pytest.mark.hardware
def test_start_daq_can_publish_processed_lsl_without_internal_recording():
    dut, threads, device, calls = _make_device_api()

    dut.start_daq(
        sampling_rate=500.0,
        do_record=False,
        do_plot=False,
        do_process=True,
    )

    assert [name for name, _, _ in threads.registered] == ["lsl_stream_system", "lsl_process_stream"]
    assert threads.registered[0][1][0] == 0
    assert [stream.name for stream in threads.registered[0][1][1]] == ["data0"]
    assert threads.registered[0][2]["require_consumers"] is True
    assert threads.registered[1][1][0:3] == (1, "data0", "filt")
    assert threads.registered[1][2]["require_consumers"] is False
    assert threads.started is True
    assert device.timeout == 2 / 500.0
    assert calls[-1] == ("write", Commands.START_DAQ, 0)


def test_start_daq_with_c_reader_registers_the_c_stream_thread():
    dut, threads, device, calls = _make_device_api()
    dut._num_skipped_bytes = 0

    dut.start_daq(
        sampling_rate=500.0, do_record=False, do_plot=False, do_process=False, use_c_reader=True
    )

    assert [name for name, _, _ in threads.registered] == ["lsl_stream_system_c"]
    args, kwargs = threads.registered[0][1], threads.registered[0][2]
    assert args[0] == 0
    assert [stream.name for stream in args[1]] == ["data0"]
    assert args[2] == [_make_daq_config()]
    assert args[3] == 42
    assert kwargs["require_consumers"] is False
    assert kwargs["on_finished"] == dut._store_c_reader_stats
    assert calls[-1] == ("write", Commands.START_DAQ, 0)


def test_c_reader_statistics_are_stored_for_stop_daq():
    dut, *_ = _make_device_api()
    dut._num_skipped_bytes = 0

    dut._store_c_reader_stats(DaqReaderStats(10, 3, 7, 0, 0, False))

    assert dut._num_package_loss == 3
    assert dut._num_skipped_bytes == 7


@pytest.mark.hardware
def test_start_daq_can_publish_raw_lsl_without_internal_recording_or_processing():
    dut, threads, _, _ = _make_device_api()

    dut.start_daq(
        sampling_rate=500.0,
        do_record=False,
        do_plot=False,
        do_process=False,
    )

    assert [name for name, _, _ in threads.registered] == ["lsl_stream_system"]
    assert threads.registered[0][1][0] == 0
    assert [stream.name for stream in threads.registered[0][1][1]] == ["data0"]
    assert threads.registered[0][2]["require_consumers"] is False


def test_start_daq_without_plot_definition_uses_default_stream_metadata():
    dut, threads, _, _ = _make_device_api()

    dut.start_daq(sampling_rate=500.0, do_plot=False, do_record=False, do_process=True)

    stream = threads.registered[0][1][1][0]
    assert (stream.channel_labels, stream.channel_units, stream.channel_layout) == (
        ["CH0", "CH1"],
        ["LSB", "LSB"],
        [[0], [1]],
    )
    kwargs = threads.registered[1][2]
    assert kwargs["channel_labels"] == ["CH0", "CH1"]
    assert kwargs["channel_units"] == ["LSB", "LSB"]
    assert kwargs["channel_layout"] == [[0], [1]]


def test_start_daq_uses_plot_definition_for_stream_metadata():
    dut, threads, _, _ = _make_device_api()
    dut.setup_live_plotter(plot_titles=[["Raw", "Filtered"]])
    dut.add_plot_channel("data0", "Voltage A", "mV", "r", (0, 0), 0)
    dut.add_plot_channel("data0", "Voltage B", "uV", "b", (0, 0), 1)
    dut.add_plot_channel("filt", "Voltage A filtered", "mV", "w", (0, 1), 0)

    dut.start_daq(sampling_rate=500.0, do_plot=False, do_record=False, do_process=True)

    stream = threads.registered[0][1][1][0]
    assert (stream.channel_labels, stream.channel_units, stream.channel_layout) == (
        ["Voltage A", "Voltage B"],
        ["mV", "uV"],
        [[0, 0], [0, 0]],
    )
    kwargs = threads.registered[1][2]
    assert kwargs["channel_labels"] == ["Voltage A filtered", "CH1"]
    assert kwargs["channel_units"] == ["mV", "LSB"]
    assert kwargs["channel_layout"] == [[0, 1], [1]]


def test_start_daq_rejects_plot_channel_outside_of_stream():
    dut, threads, _, _ = _make_device_api()
    dut.setup_live_plotter(plot_titles=[["Raw"]])
    dut.add_plot_channel("data0", "Voltage", "mV", "r", (0, 0), 5)

    with pytest.raises(ValueError):
        dut.start_daq(sampling_rate=500.0, do_plot=False, do_record=False, do_process=False)

    assert threads.registered == []


def test_start_daq_applies_plot_definition_to_each_of_multiple_streams():
    dut, threads, device, _ = _make_device_api()
    configs = [
        _make_daq_config(),
        DataAcquisitionConfig(
            send_batch=False,
            sampling_rate=250.0,
            num_channels=1,
            num_samples=1,
            head_cmd=0xFA,
            tail_cmd=0xBC,
            num_bytes_total=13,
            bytes_sample=2,
            is_signed=False,
        ),
    ]
    dut._get_number_daq_tasks = lambda: len(configs)
    dut.get_daq_characteristics = lambda sel_daq_tasks=0: configs[sel_daq_tasks]
    dut.setup_live_plotter(plot_titles=[["Raw", "Second"]])
    dut.add_plot_channel("data0", "Voltage A", "mV", "r", (0, 0), 1)
    dut.add_plot_channel("data1", "Current", "uA", "g", (0, 1), 0)

    dut.start_daq(sampling_rate=500.0, do_plot=False, do_record=False, do_process=False)

    streams = threads.registered[0][1][1]
    assert [stream.name for stream in streams] == ["data0", "data1"]
    assert [stream.head_cmd for stream in streams] == [0xA0, 0xFA]
    assert (streams[0].channel_labels, streams[0].channel_units, streams[0].channel_layout) == (
        ["CH0", "Voltage A"],
        ["LSB", "mV"],
        [[0], [0, 0]],
    )
    assert (streams[1].channel_labels, streams[1].channel_units, streams[1].channel_layout) == (
        ["Current"],
        ["uA"],
        [[0, 1]],
    )
    assert device.timeout == 2 / 250.0


def _make_process_dut():
    dut = DeviceAPI.__new__(DeviceAPI)

    class Device:
        def __init__(self) -> None:
            self.sent = []

        def convert(self, head, data):
            return data.to_bytes(3, byteorder="little") + head.to_bytes(1, byteorder="little")

        def write(self, data):
            self.sent.append(data)

    dut._device = Device()
    dut._processing_layout = ProcessingLayout(
        in_channels=2, in_samples=1, in_bytes_value=2, out_channels=1, out_samples=1, out_bytes_value=2
    )
    return dut


def _frame(data: bytes, head: Commands) -> bytes:
    return data + bytes([head])


def test_process_data_applies_value_and_triggers_without_waiting_for_response():
    dut = _make_process_dut()
    for value in (0, 1, -1, 42000, -8388608, 8388607):
        assert dut.process_data(value) is None
    sent = dut._device.sent
    assert len(sent) == 12
    assert sent[0] == _frame(b"\x00\x00\x00", Commands.APPLY_DATA)
    assert sent[1] == _frame(b"\x00\x00\x00", Commands.PROCESS_DATA)
    assert sent[4] == _frame(b"\x00\xff\xff", Commands.APPLY_DATA)
    assert sent[6] == _frame(b"\x00" + (42000).to_bytes(2, "little"), Commands.APPLY_DATA)
    assert sent[10] == _frame(b"\x00\xff\xff", Commands.APPLY_DATA)


def test_process_data_rejects_values_outside_the_data_bytes():
    dut = _make_process_dut()
    with pytest.raises(OverflowError):
        dut.process_data(8388608)
    assert dut._device.sent == []


def test_apply_data_sends_array_with_counter_in_the_order_of_the_block():
    dut = _make_process_dut()
    dut.apply_data(np.array([[1, 2], [3, 4]], dtype="<u2"))
    assert dut._device.sent == [
        _frame(b"\x00\x01\x00", Commands.APPLY_DATA),
        _frame(b"\x01\x02\x00", Commands.APPLY_DATA),
        _frame(b"\x02\x03\x00", Commands.APPLY_DATA),
        _frame(b"\x03\x04\x00", Commands.APPLY_DATA),
    ]


def test_apply_data_pads_the_last_transmission_with_zeros():
    dut = _make_process_dut()
    dut.apply_data(b"\x01\x02\x03")
    assert dut._device.sent == [
        _frame(b"\x00\x01\x02", Commands.APPLY_DATA),
        _frame(b"\x01\x03\x00", Commands.APPLY_DATA),
    ]


def test_apply_data_rejects_data_which_does_not_fit_to_the_counter():
    dut = _make_process_dut()
    dut.apply_data(bytes(2 * 256))
    assert len(dut._device.sent) == 256
    with pytest.raises(ValueError):
        dut.apply_data(bytes(2 * 256 + 1))
    assert len(dut._device.sent) == 256


def test_process_data_without_value_only_triggers_the_processing():
    dut = _make_process_dut()
    dut.process_data()
    assert dut._device.sent == [_frame(b"\x00\x00\x00", Commands.PROCESS_DATA)]


def test_process_data_applies_bytes_before_the_trigger():
    dut = _make_process_dut()
    dut.process_data(b"\x01\x02\x03\x04")
    assert dut._device.sent == [
        _frame(b"\x00\x01\x02", Commands.APPLY_DATA),
        _frame(b"\x01\x03\x04", Commands.APPLY_DATA),
        _frame(b"\x00\x00\x00", Commands.PROCESS_DATA),
    ]


def test_get_processing_layout_is_read_once():
    dut = DeviceAPI.__new__(DeviceAPI)
    calls = []

    def write_with_feedback(head, data=0, size=0):
        calls.append((head, size))
        return bytes([2, 0, 1, 0, 2, 1, 0, 1, 0, 2])

    dut._write_with_feedback = write_with_feedback
    layout = dut.get_processing_layout()
    assert dut.get_processing_layout() == layout
    assert calls == [(Commands.GET_CHARAC_PROCESSING, 11)]
    assert (layout.in_channels, layout.in_samples, layout.in_bytes_value) == (2, 1, 2)
    assert (layout.out_channels, layout.out_samples, layout.out_bytes_value) == (1, 1, 2)
    assert layout.in_block_bytes == 4
    assert layout.out_block_bytes == 2


def test_read_processing_results_returns_samples_of_the_result_stream_only():
    dut = DeviceAPI.__new__(DeviceAPI)
    dut._daq_streams = {
        0xA0: DataAcquisitionConfig(False, 100.0, 2, 1, 0xA0, 0xFF, 15, 2, False),
        0xFA: DataAcquisitionConfig(True, 100.0, 1, 2, 0xFA, 0xBC, 27, 2, False),
    }

    class Device:
        timeout = 1.0

    dut._device = Device()
    packets = [
        (0xA0, [1, 2], 1.0),
        (0xFA, [[5], [6]], [1.0, 2.0]),
        (0xFA, [[7], [8]], [3.0, 4.0]),
        (None, [], None),
    ]
    timeouts = []

    def read_packet():
        timeouts.append(dut._device.timeout)
        return packets.pop(0)

    dut._thread_read_packet = read_packet
    assert dut.read_processing_results(timeout=0.2) == [[5], [6], [7], [8]]
    assert timeouts == [0.2] * 4
    assert dut._device.timeout == 1.0


def test_start_daq_creates_default_plot_layout_file_and_applies_it(tmp_path, monkeypatch):
    path = tmp_path / "plot_config.json"
    monkeypatch.setattr("api.mcu_api.DEFAULT_PLOT_CONFIG", path)
    dut, threads, _, _ = _make_device_api()

    dut.start_daq(sampling_rate=500.0, window_sec=5.0, do_plot=True, do_record=False, do_process=True)

    assert path.exists()
    plot_config, window_length = threads.plotted
    assert window_length == 5.0
    assert plot_config is dut._plot_config
    assert [(ch.lsl_stream_name, ch.channel_name, ch.position) for ch in plot_config.channel_configs] == [
        ("data0", "CH0 Raw", (0, 0)),
        ("data0", "CH0 Temp", (1, 1)),
        ("data1", "CH1 Raw", (0, 1)),
        ("filt", "CH0 Filtered", (0, 0)),
    ]
    assert plot_config.plot_title == [["plot1", "plot2"], [None, "plot3"]]
    assert threads.registered[0][1][1][0].channel_labels == ["CH0 Raw", "CH0 Temp"]


def test_start_daq_default_plot_layout_without_processing_has_no_filt_channel(tmp_path, monkeypatch):
    monkeypatch.setattr("api.mcu_api.DEFAULT_PLOT_CONFIG", tmp_path / "plot_config.json")
    dut, threads, _, _ = _make_device_api()

    dut.start_daq(sampling_rate=500.0, do_plot=True, do_record=False, do_process=False)

    assert "filt" not in [ch.lsl_stream_name for ch in threads.plotted[0].channel_configs]


def test_start_daq_applies_existing_plot_layout_file(tmp_path, monkeypatch):
    path = tmp_path / "plot_config.json"
    monkeypatch.setattr("api.mcu_api.DEFAULT_PLOT_CONFIG", path)
    dut, _, _, _ = _make_device_api()
    dut.start_daq(sampling_rate=500.0, do_plot=True, do_record=False, do_process=False)
    path.write_text(path.read_text().replace("CH0 Temp", "Temperature"), encoding="utf-8")

    dut, threads, _, _ = _make_device_api()
    dut.start_daq(sampling_rate=500.0, do_plot=True, do_record=False, do_process=False)

    plot_config, _ = threads.plotted
    assert [ch.channel_name for ch in plot_config.channel_configs] == ["CH0 Raw", "Temperature", "CH1 Raw"]
    assert threads.registered[0][1][1][0].channel_labels == ["CH0 Raw", "Temperature"]


def test_start_daq_prefers_plot_layout_defined_in_code_over_the_file(tmp_path, monkeypatch):
    path = tmp_path / "plot_config.json"
    monkeypatch.setattr("api.mcu_api.DEFAULT_PLOT_CONFIG", path)
    dut, threads, _, _ = _make_device_api()
    dut.setup_live_plotter(plot_titles=[["Raw"]])
    dut.add_plot_channel("data0", "Voltage", "mV", "r", (0, 0), 0)

    dut.start_daq(sampling_rate=500.0, do_plot=True, do_record=False, do_process=False)

    assert not path.exists()
    assert threads.plotted[0] is dut._plot_config


def test_start_daq_creates_default_plot_layout_in_the_project_folder(tmp_path, monkeypatch):
    (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    monkeypatch.chdir(tmp_path / "sub")
    dut, _, _, _ = _make_device_api()

    dut.start_daq(sampling_rate=500.0, do_plot=True, do_record=False, do_process=False)

    assert (tmp_path / "plot_config.json").exists()
    assert not (tmp_path / "sub" / "plot_config.json").exists()
