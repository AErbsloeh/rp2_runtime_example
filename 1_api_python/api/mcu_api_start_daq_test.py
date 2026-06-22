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
    dut._DeviceAPI__layout_channels = [0, 1]
    dut._DeviceAPI__layout_labels = ["CH0", "CH1"]
    dut._DeviceAPI__num_package_loss = 0

    dut._enable_batch_daq = lambda use_batches: calls.append(("batch", use_batches))
    dut._update_daq_sampling_rate = lambda sampling_rate: calls.append(("rate", sampling_rate))
    dut.get_daq_characteristics = _make_daq_config
    dut._write_without_feedback = lambda head, data=0: calls.append(("write", head, data))

    return dut, threads, device, calls


@pytest.mark.hardware
def test_start_daq_can_publish_processed_lsl_without_internal_recording():
    dut, threads, device, calls = _make_device_api()

    dut.start_daq(
        sampling_rate=500.0,
        do_record=False,
        do_process=True,
    )

    assert [name for name, _, _ in threads.registered] == ["lsl_stream_system", "lsl_process_stream"]
    assert threads.registered[0][1][0:2] == (0, "data")
    assert threads.registered[0][2]["require_consumers"] is True
    assert threads.registered[1][1][0:3] == (1, "data", "filt")
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
        do_process=False,
    )

    assert [name for name, _, _ in threads.registered] == ["lsl_stream_system"]
    assert threads.registered[0][1][0:2] == (0, "data")
    assert threads.registered[0][2]["require_consumers"] is False
