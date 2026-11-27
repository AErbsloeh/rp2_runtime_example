# FPGA Design Folder
This folder is the place for the design of the FPGA that is connected to the MCU: HDL sources, constraints, build scripts and the generated bitstreams. The firmware (`0_firmware`) and the Python API (`2_api`) program and control the FPGA: SPI flash of the FPGA, power, program and reset lines and an SPI middleware for data exchange.

## Enable the FPGA Support
The runtime can program and control an FPGA connected to the MCU: SPI flash of the FPGA, power, program and reset lines and an SPI middleware for data exchange. The support has to be enabled on both sides:
1. **Firmware**: Enable `peri_fpga_flash` and/or `peri_fpga_spi` in `LINK_LIBS` of `../0_firmware/CMakeLists.txt`. CMake then sets `ADD_FPGA_SUPPORT` and adds `callbacks/fpga_callbacks.c`.
2. **Python**: Declare the FPGA with `DeviceAPI.has_fpga = True` before creating the interface. Without it `FlashFPGA()` raises a `RuntimeError` and the FPGA tests are skipped.

## Environment
The `pyproject.toml` of this folder loads `elasticai-hw-measurements` (from GitHub). Create the environment with `uv sync` (or open the folder with `devenv shell`), then the commands below run in this folder.

## Design Templates
Templates for FPGA designs (top module, SPI middleware, constraints, build scripts) are generated with `elasticai.fpga_testing` from the package `elasticai-hw-measurements`:
```python
from elasticai.fpga_testing import copy_design, copy_oss_cad_scripts, init_project_folder

init_project_folder()                     # once: folder structure of the FPGA workflow in your project
copy_oss_cad_scripts()                    # build scripts and, if missing, Makefile, fpga.nix and devenv files (existing ones are kept)
copy_design.copy_design_env5_files()      # design files for elasticAI.ENv5 (default destination: folder `designs`)
```
Other targets are `copy_design_arty7_files` and `copy_design_env6mini_files` (the design of the GitHub workflow). The functions also copy the cocotb testbenches of the board (`cocotb_funcs.py`, `*_echo_bench.py`, `*_led_bench.py` and the runner `<board>_test.py`) into the folder `tests`. `copy_design.copy_skeleton(...)` copies skeleton modules (dnn, filter, rom, ram, math_mult, math_mac) for own designs. The frame of the SPI middleware in the design has to match `FlashFPGA._num_spi_middleware` (command, address, data = 2, 6, 16 bit).

## Bitstream in the GitHub Workflow
The workflow `fpga.yml` builds the bitstream for the `DEVICE` of the `Makefile` (the Makefile is created from the template if it is not part of the repository, default: Lattice iCE40 UP5K).
- If the folder `designs` contains Verilog files (`*.v`), the workflow builds exactly this design. It needs the top module `TOP_MODULE` and the constraint file (`io.pcf`).
- If the folder `designs` is empty, the workflow copies the default design of the elasticAI.env6-mini (`copy_design_env6mini_files()`) and builds it.

## Simulation with cocotb and pytest
Verilog modules are simulated with [cocotb](https://www.cocotb.org/) and Icarus Verilog (`iverilog`, part of the devenv; otherwise install it or use the OSS CAD suite). The simulations are run by pytest through the fixture `cocotb_test_fixture` of `elasticai.creator`, which `conftest.py` enables via `pytest_plugins`.
```bash
uv run pytest -m simulation
```
A simulation consists of a cocotb testbench (`@cocotb.test()` with `@eai_testbench`) and a pytest function marked with `@pytest.mark.simulation`. It passes test data with `cocotb_test_fixture.write(...)`, sets the top module and the sources and starts the simulation with `run(params=..., defines=...)`; see `tests/skeleton_echo_test.py` for the echo skeleton. Every parameter set of a test ends up in its own folder in `build_test/` (ignored by git) including the waveform files (`.vcd`).
