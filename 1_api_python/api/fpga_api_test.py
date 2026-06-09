import pytest
from random import randint
from shutil import rmtree
from time import sleep
from logging import basicConfig, DEBUG
from api import get_path_to_project
from api.fpga_api import FlashFPGA


@pytest.fixture(scope="session", autouse=True)
def dut():
    FlashFPGA().do_reset()
    mcu_api = FlashFPGA(
        com_name="AUTOCOM"
    )
    mcu_api.open()
    yield mcu_api
    mcu_api.do_reset()
    path = get_path_to_project("temp_data")
    rmtree(path, ignore_errors=True)


def test_num_bytes(dut: FlashFPGA):
    assert dut.total_num_bytes == 3


def test_check_echo(dut: FlashFPGA):
    test_pattern = "TESTS"
    ret = dut.echo(test_pattern)
    assert ret == test_pattern
    assert len(ret) == len(test_pattern)


def test_check_power_state(dut: FlashFPGA):
    dut.set_power_state(False)
    assert "FPGA_PWR_EN" not in dut.get_state().pins

    dut.set_power_state(True)
    assert "FPGA_PWR_EN" in dut.get_state().pins

    dut.set_power_state(False)
    assert "FPGA_PWR_EN" not in dut.get_state().pins


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


def test_check_starting_address_in_range(dut: FlashFPGA):
    for _ in range(100):
        a = randint(a=0, b=2**32-1)
        dut._set_flash_starting_address(a)
        assert dut._get_flash_starting_address() == a


def test_check_starting_address_out_of_range(dut: FlashFPGA):
    stimuli = [randint(a=-100, b=0) for _ in range(10)]
    stimuli.extend([randint(a=2**32, b=2**32+100) for _ in range(10)])
    for val in stimuli:
        try:
            dut._set_flash_starting_address(val)
        except ValueError:
            assert True
        else:
            assert False


def test_read_page_from_flash(dut: FlashFPGA):
    page = 0
    page_size = dut.get_flash_infos().pagesize

    data = dut._read_page_from_flash(page, page_size)
    assert len(data) == page_size


if __name__ == "__main__":
    basicConfig(level=DEBUG)
    pytest.main([__file__])
