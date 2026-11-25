import cocotb
import pytest
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge
from elasticai.creator.testing import eai_testbench
from elasticai.fpga_testing import get_path_to_project


@cocotb.test()
@eai_testbench
async def data_register_testbench(dut, values):
    cocotb.start_soon(Clock(dut.CLK, 10, unit="ns").start())
    dut.RSTN.value = 0
    dut.DATA_IN.value = 0
    await FallingEdge(dut.CLK)
    await FallingEdge(dut.CLK)
    dut.RSTN.value = 1

    previous = 0
    for value in values:
        dut.DATA_IN.value = value
        assert int(dut.DATA_OUT.value) == previous
        await FallingEdge(dut.CLK)
        assert int(dut.DATA_OUT.value) == value
        previous = value


@pytest.mark.simulation
def test_data_register(cocotb_test_fixture):
    path2design = get_path_to_project("tests") / "data_register.v"
    cocotb_test_fixture.write({"values": [1, 42, 255, 0, 128]})
    cocotb_test_fixture.set_top_module_name("DATA_REGISTER").clear_srcs().add_srcs(path2design)
    cocotb_test_fixture.run(params={"BITWIDTH": 8}, defines={})
