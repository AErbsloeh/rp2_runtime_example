# FPGA
This folder is the place for the design of the FPGA that is connected to the MCU: HDL sources, constraints, build scripts and the generated bitstreams. The firmware (`0_firmware`) and the Python API (`2_api`) program and control the FPGA: SPI flash of the FPGA, power, program and reset lines and an SPI middleware for data exchange.

## Enable the FPGA Support
The runtime can program and control an FPGA connected to the MCU: SPI flash of the FPGA, power, program and reset lines and an SPI middleware for data exchange. The support has to be enabled on both sides:
1. **Firmware**: Enable `peri_fpga_flash` and/or `peri_fpga_spi` in `LINK_LIBS` of `../0_firmware/CMakeLists.txt`. CMake then sets `ADD_FPGA_SUPPORT` and adds `callbacks/fpga_callbacks.c`.
2. **Python**: Declare the FPGA with `DeviceAPI.has_fpga = True` before creating the interface. Without it `FlashFPGA()` raises a `RuntimeError` and the FPGA tests are skipped.

## Design Templates
Templates for FPGA designs (top module, SPI middleware, constraints, build scripts) are generated with `elasticai.fpga_testing` from the package `elasticai-hw-measurements`:
```python
from elasticai.fpga_testing import copy_design, copy_oss_cad_scripts, init_project_folder

init_project_folder()                     # once: folder structure of the FPGA workflow in your project
copy_oss_cad_scripts()                    # build scripts for the OSS CAD suite (e.g. Lattice iCE40UP5K, CologneChip GateMate A1)
copy_design.copy_design_env5_files()      # design files for elasticAI.env5 (default destination: folder `designs`)
```
Other targets are `copy_design_arty7_files`, `copy_design_env6mini_files` and `copy_design_olimex_gatemate_files`. `copy_design.copy_skeleton(...)` copies skeleton modules (dnn, filter, rom, ram, math_mult, math_mac) for own designs. The frame of the SPI middleware in the design has to match `FlashFPGA._num_spi_middleware` (command, address, data = 2, 6, 16 bit).

## Flashing and Control
The following code runs in the Python environment of `2_api` (`uv sync --group fpga`).
```python
from elasticai.fpga_testing.bin_build import read_bitstream_file_amd   # or read_bitstream_file_lattice
from api import DeviceAPI, FlashFPGA

DeviceAPI.has_fpga = True
dut = FlashFPGA()
bitstream = read_bitstream_file_amd(path_to_bit_file)
dut.write_bitstream_into_flash(bitstream, start_page=0, do_check=True)  # erases sectors, writes and verifies

dut.set_fpga_power_state(True)
dut.fpga_do_program_reset()
dut.fpga_do_logic_reset()
dut.fpga_toggle_led()                     # or fpga_control_led(state)
answer = dut.fpga_send_data(b"\x40\x00\x01")   # frame for the SPI middleware, returns the answer of the FPGA
```
Further functions: `get_flash_infos`, `erase_flash_all`, `erase_flash_sector`, `check_bitstream_from_flash`, `fpga_set_program_state`. `../2_api/examples/use_fpga.py` shows the complete workflow.
