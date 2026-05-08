from api import AccessFPGA, get_path_to_project
from pathlib import Path
from elasticai.fpga_testing.scripts_build.bin_build import read_bitstream_file_amd
from time import sleep

if __name__ == "__main__":
    AccessFPGA().do_reset()

    dut = AccessFPGA()
    dut.set_power_state(False)
    dut.fpga_do_reset(True)

    path2stream = Path(get_path_to_project()) / "file"
    file = [file for file in path2stream.glob("*.bit")][0]
    bitstream = read_bitstream_file_amd(file)
    if False:
        dut.write_bitstream_into_flash(
            bitstream=bitstream,
            start_page=0,
            do_check=True
        )
    print(dut.check_bitstream_from_flash(bitstream, 0))

    dut.set_power_state(True)
    for _ in range(10):
        dut.fpga_do_reset(1)
        for _ in range(10):
            dut.fpga_toggle_led()
            sleep(0.5)
        dut.toggle_led()
