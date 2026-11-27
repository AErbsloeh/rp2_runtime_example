# Project runtime template using RP2xxx MCU
[![firmware](https://github.com/AErbsloeh/rp2_runtime_example/actions/workflows/firmware.yml/badge.svg)](https://github.com/AErbsloeh/rp2_runtime_example/actions/workflows/firmware.yml)
[![api](https://github.com/AErbsloeh/rp2_runtime_example/actions/workflows/api.yml/badge.svg)](https://github.com/AErbsloeh/rp2_runtime_example/actions/workflows/api.yml)
[![fpga](https://github.com/AErbsloeh/rp2_runtime_example/actions/workflows/fpga.yml/badge.svg)](https://github.com/AErbsloeh/rp2_runtime_example/actions/workflows/fpga.yml)

This repo contains a template for projects with an RP2xxx MCU: a firmware (folder structure, basic functionality, ...), a Python API on the host computer and optionally the design of an FPGA which is connected to the MCU.
The used pins are specified to the Pico 1/2 (W). But using the default LED, there is a hardware abstraction layer which uses the default LED for different boards (Pico x, KB2040, Tiny2040).
You can use this template for building your new projects. If you have generated the template, you can move the necessary files from the library into the firmware and adapt it and also the Python API.
There is also a template for pushing sensor data in continuous manner from the MCU to a LSL stream on the host computer for real-time processing, saving in *.h5 file and for real-time plotting.

Enjoy. If you have questions and suggestions, please create an issue or contact the authors.

## Repository Structure
| Folder                  | Content                                                                                                                      |
|-------------------------|------------------------------------------------------------------------------------------------------------------------------|
| [`0_firmware`](0_firmware/README.md) | C firmware for the MCU (Pico SDK, CMake): state machine, USB or WiFi transport, DAQ streams, commands of the API |
| [`1_fpga`](1_fpga/README.md)         | Optional: design of the FPGA connected to the MCU (Verilog, build scripts, cocotb simulations with pytest)        |
| [`2_api`](2_api/README.md)           | Python API (`DeviceAPI`) with LSL streaming, live plot, HDF5 recording, C reader and digital twin                |

## Template of a Firmware
This template provides a basic structure with timer IRQ, USB handling, and state machine. The sensors will be automatic downloaded and included from [here](https://github.com/AErbsloeh/rp2_runtime_library). The content will be updated continuously and you can just update it by configuring CMake. The installation with VS Code and the Raspberry Pi Pico extension is described in [`0_firmware/README.md`](0_firmware/README.md).

## Template of a Python-API
This Python-API is usable for communicating with RP2xxx MCU using the firmware template. It is split into a generic `BasicAPI` and the application-specific `DeviceAPI` which you adapt. Installation and usage are described in [`2_api/README.md`](2_api/README.md).

## Template of an FPGA Design
The runtime can program and control an FPGA connected to the MCU. The support has to be enabled in the firmware and in the API. The designs, the build scripts for the OSS CAD suite and the simulations are described in [`1_fpga/README.md`](1_fpga/README.md).

## Continuous Integration and Releases
GitHub Actions check each folder separately and only run if files of this folder change (`.github/workflows`):

| Workflow       | Checks                                                                                    | Artifact                 |
|----------------|-------------------------------------------------------------------------------------------|--------------------------|
| `firmware.yml` | Build of the firmware for the Pico 2 (USB) with Pico SDK 2.1.1                            | `main.uf2`               |
| `api.yml`      | Linting and tests of the API (without hardware tests) and build of wheel and sdist        | wheel and sdist of `2_api` |
| `fpga.yml`     | Linting, cocotb simulations and build of the bitstream for the `DEVICE` of `1_fpga/Makefile` (default: Lattice iCE40 UP5K) | `TOP_MODULE.bin` / `.bit` |

The artifacts are attached to the workflow run (*Actions*, section *Artifacts*). Publishing a release on GitHub starts the workflows again and attaches the firmware (`firmware-pico2-<tag>.uf2`), the bitstream (`bitstream-<device>-<tag>.bin`, e.g. `bitstream-up5k-<tag>.bin`; only the bitstream is built on a release), the wheel and the sdist (`.tar.gz`, including the examples) to the release. The version of the Python package is taken from the git tag.
