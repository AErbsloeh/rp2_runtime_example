from time import sleep
from threading import Thread
import numpy as np
from pylsl import StreamOutlet, StreamInfo, cf_int16
from digital_twin import DigitalTwin, DigitalTwinConfig
from hardware_specific_functions import DACAD5765, InstrumentationAmplifier, ADCAD7779, OPA1637



class Example_data():
    def __init__(self):
        self._outlet = self._create_lsl_outlet()
        self._is_running = False


    def start(self):
        if self._is_running:
            return
        self._is_running = True
        Thread(target=self._create_testdata, daemon=True).start()


    def stop(self):
        self._is_running = False


    def _create_lsl_outlet(self):
        info = StreamInfo(name="TestStream",
                          type="data",
                          channel_count=1, 
                          nominal_srate=1000,
                          channel_format=cf_int16,
                          source_id= "TestStream_uid")
        return StreamOutlet(info)
    
    def _create_testdata(self):
        i = 0
        while self._is_running:
            test_data = np.array([i], dtype=np.int16)
            self._outlet.push_sample(test_data)
            i += 1
            sleep(0.1)


if __name__ == "__main__":
    data_generator = Example_data()
    
    # Create Object for the OPA1637 bandpass filter with a sampling frequency of 1000 Hz
    OPA1637_bandpass = OPA1637(fs=1000)

    config = DigitalTwinConfig(
        recording_name="Test",
        pipeline=[
            DACAD5765.digital_to_voltage,
            InstrumentationAmplifier.amplify_signal,
            OPA1637_bandpass.bandpass_filter,
            ADCAD7779.voltage_to_digital_two_complement
        ],
        lsl_layer_name_datasource="TestStream",
        measurement_duration=15,
        save_h5=True
    )
    digital_twin = DigitalTwin(config)
    
    data_generator.start()
    digital_twin.start()
    sleep(config.measurement_duration)
    digital_twin.stop()
    data_generator.stop()