import os

from api.src._interface_serial import InterfaceSerial


def test_timeout_can_be_read_and_changed():
    master, slave = os.openpty()
    try:
        dut = InterfaceSerial(os.ttyname(slave), timeout=1.0)

        assert dut.timeout == 1.0
        dut.timeout = 0.05
        assert dut.timeout == 0.05
        assert dut.read(1) == b""
        dut.timeout = 1.0
        assert dut.timeout == 1.0
        dut.close()
    finally:
        os.close(master)
        os.close(slave)
