import binascii
from logging import getLogger

import pytest

from api.definitions import Commands, DaqDataType
from api.src._basic_api import BasicAPI
from api.src._helper import DataAcquisitionConfig


def test_echo_value_sends_and_decodes_signed_values():
    dut = BasicAPI.__new__(BasicAPI)
    sent = []

    def fake_write_with_feedback(head, data=0, size=0):
        sent.append((head, data))
        return data.to_bytes(3, byteorder="big")

    dut._write_with_feedback = fake_write_with_feedback

    for value in (0, 1, -1, 42000, -8388608, 8388607):
        assert dut.echo_value(value) == value
    assert all(head == Commands.ECHO for head, _ in sent)
    assert sent[2][1] == 0xFFFFFF


def test_echo_value_rejects_values_outside_the_data_bytes():
    dut = BasicAPI.__new__(BasicAPI)
    dut._write_with_feedback = lambda head, data=0, size=0: data.to_bytes(3, byteorder="big")
    with pytest.raises(OverflowError):
        dut.echo_value(8388608)


class FakeReadDevice:
    def __init__(self) -> None:
        self.feed = bytearray()

    def read(self, num_bytes: int) -> bytes:
        chunk = bytes(self.feed[:num_bytes])
        del self.feed[:num_bytes]
        return chunk


def make_packet(
    config: DataAcquisitionConfig, index: int, timestamps: list[int], values: list[int]
) -> bytes:
    data = bytes([config.head_cmd, index])
    for timestamp in timestamps:
        data += timestamp.to_bytes(8, "little")
    for value in values:
        data += value.to_bytes(config.bytes_sample, "little", signed=config.is_signed)
    return data + binascii.crc_hqx(data, 0xFFFF).to_bytes(2, "little") + bytes([config.tail_cmd])


def make_reader_api(configs: list[DataAcquisitionConfig]) -> tuple[BasicAPI, FakeReadDevice]:
    dut = BasicAPI.__new__(BasicAPI)
    dut._device = FakeReadDevice()
    dut._logger = getLogger(__name__)
    dut._num_package_loss = 0
    dut._register_daq_streams(configs)
    return dut, dut._device


CONFIG_SAMPLE = DataAcquisitionConfig(False, 100.0, 2, 1, 0xA0, 0xFF, 3 + 8 + 4 + 2, 2, True)
CONFIG_BATCH = DataAcquisitionConfig(True, 100.0, 1, 4, 0xB0, 0xFF, 3 + 16 + 4 * 2 + 2, 2, False)


def test_read_packets_of_two_streams_with_garbage_in_between():
    dut, device = make_reader_api([CONFIG_SAMPLE, CONFIG_BATCH])
    device.feed += b"\x00\x13" + make_packet(CONFIG_SAMPLE, 0, [1_000_000], [-5, 7])
    device.feed += make_packet(CONFIG_BATCH, 0, [2_000_000, 2_030_000], [1, 2, 3, 4])
    bad = bytearray(make_packet(CONFIG_SAMPLE, 1, [1_010_000], [9, 9]))
    bad[5] ^= 0xFF
    device.feed += bytes(bad) + make_packet(CONFIG_SAMPLE, 2, [1_020_000], [3, -3])

    head, data, timestamp = dut._thread_read_packet()
    assert (head, data, timestamp) == (0xA0, [-5, 7], pytest.approx(1.0))
    head, data, timestamps = dut._thread_read_packet()
    assert head == 0xB0
    assert data == [[1], [2], [3], [4]]
    assert timestamps == pytest.approx([2.0, 2.01, 2.02, 2.03])
    head, data, timestamp = dut._thread_read_packet()
    assert (head, data, timestamp) == (0xA0, [3, -3], pytest.approx(1.02))
    assert dut._num_skipped_bytes >= 2 + len(bad)
    assert dut._num_package_loss == 1


def test_read_returns_nothing_on_timeout():
    dut, _ = make_reader_api([CONFIG_SAMPLE])
    assert dut._thread_read_packet() == (None, [], None)


def test_register_rejects_duplicate_head_bytes():
    with pytest.raises(ValueError):
        make_reader_api([CONFIG_SAMPLE, CONFIG_SAMPLE])


CONFIG_24BIT_SIGNED = DataAcquisitionConfig(False, 100.0, 2, 1, 0xFB, 0xBD, 3 + 8 + 6 + 2, 3, True)
CONFIG_24BIT_BATCH = DataAcquisitionConfig(True, 100.0, 1, 3, 0xFC, 0xBE, 3 + 16 + 9 + 2, 3, False)


def test_read_packets_with_24_bit_values():
    dut, device = make_reader_api([CONFIG_24BIT_SIGNED, CONFIG_24BIT_BATCH])
    device.feed += make_packet(CONFIG_24BIT_SIGNED, 0, [1_000_000], [-8388608, 8388607])
    device.feed += make_packet(CONFIG_24BIT_SIGNED, 1, [1_010_000], [-1, 1234])
    device.feed += make_packet(CONFIG_24BIT_BATCH, 0, [2_000_000, 2_020_000], [0, 8388608, 16777215])

    head, data, timestamp = dut._thread_read_packet()
    assert (head, data, timestamp) == (0xFB, [-8388608, 8388607], pytest.approx(1.0))
    head, data, _ = dut._thread_read_packet()
    assert (head, data) == (0xFB, [-1, 1234])
    head, data, timestamps = dut._thread_read_packet()
    assert head == 0xFC
    assert data == [[0], [8388608], [16777215]]
    assert timestamps == pytest.approx([2.0, 2.01, 2.02])


def _charac_frame(data_type: int) -> bytes:
    return (
        bytes([Commands.GET_CHARAC_DAQ, 0xFB, 0xBD, 1, 0])
        + (1).to_bytes(2, "little")
        + (1).to_bytes(2, "little")
        + (3).to_bytes(2, "little")
        + bytes([3, data_type])
        + (10000).to_bytes(8, "little")
    )


@pytest.mark.parametrize("data_type", list(DaqDataType))
def test_get_daq_characteristics_contains_data_type(data_type):
    dut = BasicAPI.__new__(BasicAPI)
    dut._num_daq_tasks = 1
    sizes = []

    class Device:
        def convert(self, head, data):
            return bytes([head, data])

        def write_wfb(self, data, size=0):
            sizes.append(size)
            return _charac_frame(data_type)

    dut._device = Device()
    config = dut.get_daq_characteristics(0)
    assert sizes == [21]
    assert config.data_type == data_type
    assert (config.head_cmd, config.bytes_sample, config.is_signed) == (0xFB, 3, True)
    assert config.sampling_rate == pytest.approx(100.0)


def test_data_type_defaults_to_sample():
    assert CONFIG_SAMPLE.data_type == DaqDataType.SAMPLE


def test_init_stops_a_daq_that_still_runs_from_a_previous_session(monkeypatch):
    from api.mcu_api import DeviceAPI

    class FakeSerial:
        def __init__(self, *args, **kwargs) -> None:
            self.events = []
            self.pending = 100
            self.is_open = False

        def open(self) -> None:
            self.events.append("open")
            self.is_open = True

        def close(self) -> None:
            self.is_open = False

        def convert(self, head: int, data: int) -> bytes:
            return bytes([head])

        def write(self, data: bytes) -> None:
            self.events.append(("write", data[0]))
            if data[0] == Commands.STOP_DAQ:
                self.pending = 0

        def empty_buffer(self) -> int:
            num, self.pending = self.pending, 0
            self.events.append("empty")
            return num

    monkeypatch.setattr("api.src._basic_api.InterfaceSerial", FakeSerial)
    monkeypatch.setattr("api.src._basic_api.sleep", lambda _: None)

    dut = DeviceAPI(com_name="COM_TEST")

    assert dut._device.events[0] == "open"
    assert ("write", int(Commands.STOP_DAQ)) in dut._device.events
    assert dut._device.events[-1] == "empty"
