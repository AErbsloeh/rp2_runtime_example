from logging import INFO, basicConfig
from threading import Event, Thread
from time import monotonic, sleep

import numpy as np
from hardware_specific_functions import ADCAD7779, DACAD5765, OPA1637, InstrumentationAmplifier
from pylsl import StreamInfo, StreamOutlet, cf_int32

from twin import DigitalTwin, DigitalTwinConfig


class ExampleData:
    """Generating test data (DAC codes) in a LSL stream with the nominal sampling rate"""

    def __init__(self, name: str = "TestStream", sampling_rate: float = 1000.0) -> None:
        self._fs = sampling_rate
        self._stop = Event()
        self._thread: Thread | None = None
        info = StreamInfo(
            name=name,
            type="data",
            channel_count=1,
            nominal_srate=sampling_rate,
            channel_format=cf_int32,
            source_id=f"{name}_uid",
        )
        info.set_channel_labels(["DAC code"])
        info.set_channel_units(["LSB"])
        info.set_channel_types(["data"])
        self._outlet = StreamOutlet(info)

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = Thread(target=self._create_testdata, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    def _create_testdata(self) -> None:
        # 5 Hz sine + small 100 Hz part and an offset to the mid code of the DAC, the bandpass of the twin removes the offset
        t_start = monotonic()
        num_sent = 0
        while not self._stop.is_set():
            num_due = int((monotonic() - t_start) * self._fs) - num_sent
            if num_due > 0:
                t = (num_sent + np.arange(num_due)) / self._fs
                codes = 32768 + 4000 + 10000 * np.sin(2 * np.pi * 5 * t) + 2000 * np.sin(2 * np.pi * 100 * t)
                self._outlet.push_chunk(codes.astype(np.int32)[:, np.newaxis].tolist())
                num_sent += num_due
            sleep(0.005)


if __name__ == "__main__":
    basicConfig(level=INFO)
    do_plot = True
    sampling_rate = 1000.0

    data_generator = ExampleData(sampling_rate=sampling_rate)

    OPA1637_bandpass = OPA1637(fs=sampling_rate, number_of_channels=1)

    config = DigitalTwinConfig(
        pipeline=[
            DACAD5765.digital_to_voltage,
            InstrumentationAmplifier.amplify_signal,
            OPA1637_bandpass.bandpass_filter,
            ADCAD7779.voltage_to_digital_two_complement,
        ],
        lsl_layer_name_datasource="TestStream",
        save_h5=True,
    )
    digital_twin = DigitalTwin(config)

    data_generator.start()
    digital_twin.start()
    if do_plot:
        digital_twin.plot(window_sec=5.0)
    else:
        digital_twin.wait(15)
    digital_twin.stop()
    data_generator.stop()
