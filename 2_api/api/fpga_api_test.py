from logging import DEBUG, basicConfig
from random import randint
from types import SimpleNamespace

import pytest

from api import BitstreamVendor, DeviceAPI
from api.fpga_api import FlashFPGA

VENDOR = "amd"
PAYLOAD = b"\xff" * 32 + b"\x00\x00\x00\xbb\x11\x22\x00\x44" + b"\xaa\x99\x55\x66" + bytes(range(64))
HEADER = b"\x00\x09\x0f\xf0\x0f\xf0\x0f\xf0\x0f\xf0\x00\x01a\x00\x0adesign;UTF\x00"


@pytest.fixture
def fake_dut(monkeypatch):
    fake_dut = object.__new__(FlashFPGA)
    fake_dut._fpga_vendor = BitstreamVendor[VENDOR.upper()]
    written = []
    sets = SimpleNamespace(pagesize=256, num_pages=1024, blocksize=16)
    monkeypatch.setattr(fake_dut, "get_flash_infos", lambda: sets, raising=False)
    monkeypatch.setattr(fake_dut, "set_fpga_power_state", lambda state: None, raising=False)
    monkeypatch.setattr(fake_dut, "fpga_set_program_state", lambda state: None, raising=False)
    monkeypatch.setattr(fake_dut, "get_state", lambda: SimpleNamespace(pins=[]), raising=False)
    monkeypatch.setattr(fake_dut, "erase_flash_sector", lambda page, size: None, raising=False)
    monkeypatch.setattr(
        fake_dut,
        "_write_page_into_flash",
        lambda page, data, size, check: written.append((page, bytes(data))),
        raising=False,
    )
    fake_dut.written = written
    return fake_dut


def _stream(written: list) -> bytes:
    return b"".join(data for _, data in written)


def test_write_reads_the_bit_file_of_the_vendor(fake_dut, tmp_path):
    path = tmp_path / "design.bit"
    path.write_bytes(HEADER + PAYLOAD)

    fake_dut.write_bitstream_into_flash(path)

    assert _stream(fake_dut.written).startswith(PAYLOAD)
    assert HEADER not in _stream(fake_dut.written)


def test_write_uses_the_vendor_of_the_device(fake_dut, tmp_path):
    path = tmp_path / "design.bin"
    path.write_bytes(PAYLOAD)
    fake_dut._fpga_vendor = BitstreamVendor.LATTICE

    fake_dut.write_bitstream_into_flash(path)

    assert _stream(fake_dut.written).startswith(PAYLOAD)


def test_write_rejects_wrong_file_before_touching_the_flash(fake_dut, tmp_path):
    path = tmp_path / "design.cfg"
    path.write_text("D9 01 ED\n")
    fake_dut._fpga_vendor = BitstreamVendor.CCGM

    with pytest.raises(ValueError):
        fake_dut.write_bitstream_into_flash(path)
    assert fake_dut.written == []


def test_check_uses_the_same_reader(fake_dut, tmp_path, monkeypatch):
    path = tmp_path / "design.bit"
    path.write_bytes(HEADER + PAYLOAD)
    pages = [PAYLOAD[i : i + 256].ljust(256, b"\xff") for i in range(0, len(PAYLOAD), 256)]
    monkeypatch.setattr(fake_dut, "_read_page_from_flash", lambda page, size: pages[page], raising=False)

    assert fake_dut.check_bitstream_from_flash(path, 0)


def test_vendor_can_be_given_as_enum_or_as_lower_case_name(monkeypatch):
    monkeypatch.setattr("api.fpga_api.DeviceAPI.__init__", lambda self, *args, **kwargs: None)
    monkeypatch.setattr("api.fpga_api.FlashFPGA._FlashFPGA__init_fpga", lambda self: None)
    monkeypatch.setattr("api.fpga_api.FlashFPGA.extend_pins_list", lambda self, pins: None, raising=False)
    monkeypatch.setattr(
        "api.fpga_api.FlashFPGA.extend_states_list", lambda self, states: None, raising=False
    )
    monkeypatch.setattr("api.fpga_api.FlashFPGA.set_fpga_power_state", lambda self, state: None)
    monkeypatch.setattr("api.fpga_api.FlashFPGA.fpga_set_program_state", lambda self, state: None)
    monkeypatch.setattr("api.fpga_api.DeviceAPI.has_fpga", True)

    assert FlashFPGA("ccgm")._fpga_vendor == BitstreamVendor.CCGM
    assert FlashFPGA("AMD")._fpga_vendor == BitstreamVendor.AMD
    assert FlashFPGA(BitstreamVendor.LATTICE)._fpga_vendor == BitstreamVendor.LATTICE


def test_fpga_is_not_declared_by_default():
    assert DeviceAPI.has_fpga is False


def test_flash_fpga_requires_declared_fpga(monkeypatch):
    monkeypatch.setattr(DeviceAPI, "has_fpga", False)

    with pytest.raises(RuntimeError, match="has_fpga"):
        FlashFPGA(vendor=VENDOR)


@pytest.fixture(scope="module")
def dut():
    if not DeviceAPI.has_fpga:
        pytest.skip("No FPGA declared (DeviceAPI.has_fpga is False)")
    try:
        FlashFPGA(vendor=VENDOR).do_reset()
        mcu_api = FlashFPGA(vendor=VENDOR, com_name="AUTOCOM")
        mcu_api.open()
    except ConnectionError as error:
        pytest.skip(f"No device with FPGA connected: {error}")
    yield mcu_api
    mcu_api.do_reset()


@pytest.mark.fpga
@pytest.mark.hardware
def test_num_bytes(dut: FlashFPGA):
    assert dut.total_num_bytes == dut._num_bytes_head + dut._num_bytes_data


@pytest.mark.fpga
@pytest.mark.hardware
def test_check_echo(dut: FlashFPGA):
    test_pattern = "TESTS"
    ret = dut.echo(test_pattern)
    assert ret == test_pattern
    assert len(ret) == len(test_pattern)


@pytest.mark.fpga
@pytest.mark.hardware
def test_check_power_state(dut: FlashFPGA):
    dut.set_fpga_power_state(False)
    assert "FPGA_PWR_EN" not in dut.get_state().pins

    dut.set_fpga_power_state(True)
    assert "FPGA_PWR_EN" in dut.get_state().pins

    dut.set_fpga_power_state(False)
    assert "FPGA_PWR_EN" not in dut.get_state().pins


@pytest.mark.fpga
@pytest.mark.hardware
def test_check_flash_infos(dut: FlashFPGA):
    result = dut.get_flash_infos()
    assert result.manu_id == 1
    assert result.dev_id == 24
    assert result.mem_type == 2
    assert result.capacity == 33554432
    assert result.status == 0
    assert result.pagesize == 256
    assert result.num_pages == 131072
    assert result.num_blocks == 1024


@pytest.mark.fpga
@pytest.mark.hardware
def test_check_starting_address_in_range(dut: FlashFPGA):
    for _ in range(100):
        a = randint(a=0, b=2**32 - 1)
        dut._set_flash_starting_address(a)
        assert dut._get_flash_starting_address() == a


@pytest.mark.fpga
@pytest.mark.hardware
def test_check_starting_address_out_of_range(dut: FlashFPGA):
    stimuli = [randint(a=-100, b=0) for _ in range(10)]
    stimuli.extend([randint(a=2**32, b=2**32 + 100) for _ in range(10)])
    for val in stimuli:
        try:
            dut._set_flash_starting_address(val)
        except ValueError:
            assert True
        else:
            assert False


@pytest.mark.fpga
@pytest.mark.hardware
def test_read_page_from_flash(dut: FlashFPGA):
    page = 0
    page_size = dut.get_flash_infos().pagesize

    data = dut._read_page_from_flash(page, page_size)
    assert len(data) == page_size


if __name__ == "__main__":
    basicConfig(level=DEBUG)
    pytest.main([__file__])
