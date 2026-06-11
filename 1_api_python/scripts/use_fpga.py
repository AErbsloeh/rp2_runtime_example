from api import FlashFPGA, get_path_to_project
from pathlib import Path
from elasticai.fpga_testing.scripts_build.bin_build import read_bitstream_file_amd
from time import sleep


if __name__ == "__main__":
    do_flash = True
    do_erase = False
    FlashFPGA().do_reset()

    dut = FlashFPGA()
    if do_flash:
        path2stream = Path(get_path_to_project()) / "file"
        if not path2stream.exists():
            raise FileNotFoundError()

        file = [file for file in path2stream.glob("*.bit")][0]
        bitstream = read_bitstream_file_amd(file)
        if do_erase:
            dut.erase_flash_all()

        dut.write_bitstream_into_flash(
            bitstream=bitstream,
            start_page=0,
            do_check=True
        )
        if not dut.check_bitstream_from_flash(bitstream, 0):
            raise ValueError("Bitstream is not equal to the content of the FPGA flash!")

    dut.set_power_state(True)
    dut.fpga_do_program_reset()
    sleep(2)
    dut.fpga_do_logic_reset()
    for _ in range(10):
        for _ in range(20):
            dut.fpga_toggle_led()
            sleep(0.5)
        dut.toggle_led()
