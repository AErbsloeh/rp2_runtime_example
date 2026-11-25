import pytest

from api import DeviceAPI
from api.fpga_api import FlashFPGA


def test_fpga_is_not_declared_by_default():
    assert DeviceAPI.has_fpga is False


def test_flash_fpga_requires_declared_fpga(monkeypatch):
    monkeypatch.setattr(DeviceAPI, "has_fpga", False)

    with pytest.raises(RuntimeError, match="has_fpga"):
        FlashFPGA()
