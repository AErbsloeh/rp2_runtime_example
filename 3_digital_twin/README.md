# 3_digital_twin

This module provides a framework for a "Digital Twin". It enables real-time processing and recording of sensor data streams via the Lab Streaming Layer (LSL) using a customizable processing pipeline.

## Key Features

- **LSL Integration**: Receives real-time data from a specified LSL (input) stream and pushes the processed results directly to a new LSL (output) stream named `DigitalTwinOutput`.
- **Customizable Data Pipeline**: Data can be processed, filtered, and transformed through a sequence of user-defined callable functions.
- **HDF5 Recording**: If configured, processed data is saved into `.h5` files, managed by the `H5Handler`.

## File Structure & Core Components

- `digital_twin.py`: Contains the main class `DigitalTwin` (stream and thread management) and its configuration dataclass `DigitalTwinConfig`.
- `lsl_handler.py`: Module for connecting to (`StreamInlet`) and creating (`StreamOutlet`) LSL streams.
- `h5_handler.py`: Manages the storage of signals into structured HDF5 files.
- `hardware_specific_functions.py`: A collection of classes that represent specific hardware characteristics.
Additional specific hardware characteristics can be added here.
- `example.py`: A basic example script demonstrating how to configure and deploy the Digital Twin.
- `pyproject.toml`: Package and dependency configurations for the project.

## Usage

To use the Digital Twin, define a configuration object and pass it to the twin.

The basic function is shown in the example.py file.


It is recommended to use an environment based on the provided `pyproject.toml`.
