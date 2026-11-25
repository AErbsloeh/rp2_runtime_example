from api import FlashFPGA, get_path_to_project
from pathlib import Path
from elasticai.fpga_testing.bin_build import read_bitstream_file_amd
from time import sleep


if __name__ == "__main__":
    do_flash = True
    do_erase = False
    FlashFPGA().do_reset()

    dut = FlashFPGA()
    if do_flash:
        path2stream = Path(get_path_to_project()).parent / "1_fpga" / "build"
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

    dut.set_fpga_power_state(True)
    dut.fpga_do_program_reset()
    dut.fpga_do_logic_reset()
    sleep(2)
    print(dut.get_state())

    old_val = -1
    for _ in range(100):
        dut.fpga_toggle_led()
        for _ in range(4):
            sleep(0.1)
            val = int(dut.fpga_send_data(b"\x40\x00\x01").hex(), 16)
            print(val)
            assert val - old_val == 1
            old_val = val
