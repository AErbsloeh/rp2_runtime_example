from logging import INFO, basicConfig

import numpy as np
from twin_config import FirmwareProcessing

from api import DeviceAPI
from twin import DeviceLoopback, DigitalTwin, DigitalTwinConfig, ErrorMonitor, LslGenerator


def dac_codes(t: np.ndarray) -> np.ndarray:
    """Test signal as DAC codes: 5 Hz sine with a small 100 Hz part and an offset"""
    return 32768 + 4000 + 10000 * np.sin(2 * np.pi * 5 * t) + 2000 * np.sin(2 * np.pi * 100 * t)


if __name__ == "__main__":
    basicConfig(level=INFO)
    DeviceAPI().do_reset()
    REALTIME = False
    SAMPLING_RATE = 500.0

    generator = LslGenerator("TestStream", SAMPLING_RATE, dac_codes, realtime=REALTIME)
    twin = DigitalTwin(
        DigitalTwinConfig(
            pipeline=[FirmwareProcessing.run_sequential],
            lsl_layer_name_datasource="TestStream",
        )
    )
    device = DeviceAPI()
    mcu = DeviceLoopback(
        process=device.process_data,
        name_in="TestStream",
        name_out="McuEcho",
        label="MCU",
        read_results=device.read_processing_results,
        realtime=REALTIME,
        generator=generator,
    )
    monitor = ErrorMonitor(
        name_device="McuEcho", name_twin=twin.output_stream_name, name_out="McuError", realtime=REALTIME
    )

    for part in (generator, twin, monitor, mcu):
        part.start()

    twin.plot(
        window_sec=10.0,
        extra_streams=[("McuEcho", "MCU", "LSB")],
        error_streams=[("McuError", "Error", "LSB")],
        title_info=lambda: f"MAE: {monitor.mae:.2f}",
    )
    for part in (mcu, monitor, twin, generator):
        part.stop()
