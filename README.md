# Project runtime template using RP2xxx MCU 
This repo contains a simple firmware template (folder structure, basic functionality, ...) for deploying projects. 
The used pins are specified to the Pico 1/2 (W). But using the default LED, there is a hardware abstraction layer which uses the default LED for different boards (Pico x, KB2040, Tiny2040).
You can use this template for building your new projects. If you have generated the template, you can move the necessary files from the library into the firmware and adapt it and also the Python API. 
There is also a template for pushing sensor data in continous manner from the MCU to a LSL stream on the host computer for real-time processing, saving in *.h5 file and for real-time plotting. 

Enjoy. If you have questions and suggestions, please create an issue or contact the authors.

## Template of a Firmware
This template provides a basic structure with timer IRQ, USB handling, and state machine. The sensors will be automatic downloaded and included from [here](https://github.com/AErbsloeh/pico_runtime_library). The content will be updateded continously and you can just update it by configuring CMake.

## Template of a Python-API
This Python-API is usable for communicating with RP2xxx MCU using the firmware template.

## Template of C-API 
This C-API is usable for data acquisition using the firmware template (future feature). 
