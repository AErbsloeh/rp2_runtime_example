# C Reader for the DAQ Streams
Library `daq_reader` for reading the DAQ packets of the firmware on the host in C: a thread reads the file descriptor of the opened interface (USB serial port or TCP socket), finds and validates the packets of all streams (head, tail, CRC-16, re-synchronisation, package-loss detection) and pushes the decoded data with the device timestamps into LSL outlets. The Python API (`2_api`) uses it with `DeviceAPI.start_daq(..., use_c_reader=True)`, so no packet is handled in Python.

The library needs neither the header nor the library of liblsl: the push functions (`lsl_push_sample_itp`, `lsl_push_chunk_itnp`) are given by the caller as function pointers together with the outlet handles. All channels are pushed as `int32`.

## Build
```bash
cmake -S . -B build
cmake --build build
```
Without CMake: `cc -std=c11 -O2 -fPIC -shared -o build/libdaq_reader.so daq_reader.c -pthread`.

The Python API finds `build/libdaq_reader.so` (or `DAQ_READER_LIB`) and compiles the library with the system C compiler on first use if it is missing. Supported are POSIX systems (Linux, macOS).

## Usage in C
```c
daq_stream_config_t config = {
    .head = 0xA0, .tail = 0xFF, .is_batch = 0, .is_signed = 1, .bytes_sample = 2, .has_crc = 1,
    .num_channels = 2, .num_samples = 1, .num_bytes_total = 3 + 8 + 2 * 2 + 2, .outlet = outlet,
};
daq_reader_t *reader = daq_reader_create(&config, 1, lsl_push_sample_itp, lsl_push_chunk_itnp);
daq_reader_start(reader, fd);
...
daq_reader_get_stats(reader, &stats);
daq_reader_stop(reader);
daq_reader_destroy(reader);
```

## Packet Layout
`head (1) | index (1) | timestamp [us] (8, batch: first and last = 16) | data (little endian, sample-major) | CRC-16/CCITT-FALSE (2, optional) | tail (1)`
