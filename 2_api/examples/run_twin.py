from logging import INFO, basicConfig

import numpy as np
from hardware_specific_functions import ADCAD7779, DACAD5765, OPA1637, InstrumentationAmplifier

from api import DeviceAPI
from twin import DeviceLoopback, DigitalTwin, DigitalTwinConfig, LslGenerator


def dac_codes(t: np.ndarray) -> np.ndarray:
    """Test signal as DAC codes: 5 Hz sine with a small 100 Hz part and an offset (removed by the bandpass of the twin)"""
    return 32768 + 4000 + 10000 * np.sin(2 * np.pi * 5 * t) + 2000 * np.sin(2 * np.pi * 100 * t)


if __name__ == "__main__":
    basicConfig(level=INFO)
    USE_MCU = True
    SAMPLING_RATE = 1000.0

    generator = LslGenerator("TestStream", SAMPLING_RATE, dac_codes)
    mcu = None
    if USE_MCU:
        device = DeviceAPI()
        mcu = DeviceLoopback(
            device.process_data,
            "TestStream",
            name_out="McuEcho",
            label="MCU",
            read_results=device.read_processing_results,
        )
    twin = DigitalTwin(
        DigitalTwinConfig(
            pipeline=[
                DACAD5765.digital_to_voltage,
                InstrumentationAmplifier.amplify_signal,
                OPA1637(fs=SAMPLING_RATE, number_of_channels=1).bandpass_filter,
                ADCAD7779.voltage_to_digital_two_complement,
            ],
            lsl_layer_name_datasource="TestStream",
        )
    )

    for part in (generator, mcu, twin):
        if part is not None:
            part.start()

    twin.plot(window_sec=5.0, extra_streams=[("McuEcho", "MCU", "LSB")] if USE_MCU else None)
    for part in (twin, mcu, generator):
        if part is not None:
            part.stop()
