from logging import getLogger

import pytest

from api.mcu_api import (
    Commands,
    DeviceAPI,
)
from api.src._helper import DataAcquisitionConfig


class FakeDevice:
    timeout: float

    def __init__(self) -> None:
        self.timeout = 0.0


class FakeThreads:
    def __init__(self) -> None:
        self.registered = []
        self.started = False

    def register(self, func, args, kwargs=None) -> None:
        self.registered.append((func.__name__, args, kwargs or {}))

    def start(self) -> None:
        self.started = True

    def lsl_stream_util(self) -> None:
        pass

    def lsl_stream_system(self) -> None:
        pass

    def lsl_process_stream(self) -> None:
        pass

    def lsl_record_stream(self) -> None:
        pass

    def lsl_plot_stream(self) -> None:
        pass


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

    dut._DeviceAPI__threads = threads
    dut._DeviceAPI__device = device
    dut._DeviceAPI__logger = getLogger(__name__)
    dut._DeviceAPI__plot_config = None
    dut._DeviceAPI__num_package_loss = 0

    dut._enable_batch_daq = lambda use_batches: calls.append(("batch", use_batches))
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
