from api import AccessFPGA, get_path_to_project
from pathlib import Path
from elasticai.fpga_testing.scripts_build.bin_build import read_bitstream_file_amd
from time import sleep

if __name__ == "__main__":
    do_flash = True
    AccessFPGA().do_reset()

    dut = AccessFPGA()
    if do_flash:
        path2stream = Path(get_path_to_project()) / "file"
        file = [file for file in path2stream.glob("*.bit")][0]
        bitstream = read_bitstream_file_amd(file)

        #dut.erase_flash_all()
        dut.write_bitstream_into_flash(
            bitstream=bitstream,
            start_page=0,
            do_check=False
        )
        print(dut.check_bitstream_from_flash(bitstream, 0))

    dut.set_power_state(True)
    dut.fpga_do_program_reset()
    sleep(2)
    dut.fpga_do_logic_reset()
    for _ in range(10):
        for _ in range(20):
            dut.fpga_toggle_led()
            sleep(0.5)
        dut.toggle_led()
