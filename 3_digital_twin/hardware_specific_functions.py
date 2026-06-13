import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi

class DACAD5765:
    @staticmethod
    def digital_to_voltage(value: np.ndarray, v_ref: float=5.) -> np.ndarray[np.float32]:
        """Convert digital values from the AD5765 DAC to voltage values using the reference voltage and the resolution of the DAC. The AD5765 is a 16-bit DAC, so the digital values are expected to be in the range of 0 to 65535 (2^16 - 1). The conversion formula is: voltage = v_ref * ((value / (2^15)) - 1), where value is the digital input and v_ref is the reference voltage.

        Args:
            value (np.ndarray): The digital values to be converted, expected to be a NumPy array of integers in the range 0 to 65535.
            v_ref (float, optional): The reference voltage in volts. Defaults to 5..

        Returns:
            np.ndarray[np.float64]: The converted voltage values, returned as a NumPy array of floating point numbers.
        """
        codes_clipped = np.clip(value, 0, 65535)
        signal_voltage = v_ref * ((codes_clipped / (2**15)) -1)
        return signal_voltage.astype(np.float32)
    

class InstrumentationAmplifier:
    @staticmethod
    def amplify_signal(signal: np.ndarray, gain: float= 1.2) -> np.ndarray[np.float32]:
        """Amplify the input signal by multiplying it with the specified gain factor

        Args:
            signal (np.ndarray): The input signal to be amplified, expected to be a NumPy array of any shape.
            gain (float): The amplification factor by which the input signal will be multiplied. The gain need to be a positive value greater than 1

        Returns:
            np.ndarray[np.float64]: The amplified signal, returned as a NumPy array of the same shape as the input signal.
        """        
        return signal * gain
    

class ADCAD7779:
    @staticmethod
    def two_complement_to_voltage(signal_values: np.ndarray, v_ref: float=2.5, pga_gain: float=1., g_extra: float=1.) -> np.ndarray[np.float32]:
        """Convert two's complement values to voltage values

        Args:
            signal_values (np.ndarray): The two's complement values to be converted
            v_ref (float, optional): The reference voltage in volts. Defaults to 2.5.
            pga_gain (float, optional): The gain of the programmable gain amplifier. Defaults to 1..
            g_extra (float, optional): The extra gain factor. Defaults to 1..

        Returns:
            np.ndarray[np.float32]: The converted voltage values
        """        
        value = ((signal_values * pga_gain * g_extra) /(v_ref * 2) * (2**24))
        return value.astype(np.float32)
    

    @staticmethod
    def voltage_to_digital_two_complement(signal_voltage: np.ndarray, v_ref: float = 2.5, pga_gain: float = 1.) -> np.ndarray[np.int32]:
        """Convert voltage values to digital values using the reference voltage, gain factors and the resolution of the ADC.

        Args:
            signal_voltage (np.ndarray): The input voltage values to be converted, expected to be a NumPy array of floating point numbers.
            v_ref (float, optional): The reference voltage in volts. Defaults to 2.5.
            pga_gain (float, optional): The gain of the programmable gain amplifier. Defaults to 1..
        """

        max_code = 8388607 #(2^23) - 1
        min_code = -8388608 #-(2^23)

        fsr = v_ref / pga_gain
        lsb_size = (2 * fsr) / (2 ** 24)
        codes = np.round(signal_voltage / lsb_size)
        codes = np.clip(codes, min_code, max_code)
        return codes.astype(np.int32)


class OPA1637:
    def __init__(self, fs: float = 1000, number_of_channels: int = 4) -> None:
        """Initialize the OPA1637 bandpass filter with the specified sampling frequency and number of channels

        Args:
            fs (float, optional): The sampling frequency in Hz. Defaults to 1000.
            number_of_channels (int, optional): The number of channels for the filter. Defaults to 4.
        """        
        self.sos = butter(N=1, Wn=[0.33, 400.0], btype='bandpass', fs=fs, output='sos')
        self.zi = []
        for i in range(number_of_channels):
            self.zi.append(sosfilt_zi(self.sos))
        self._initialized = False

    def bandpass_filter(self, signal: np.ndarray) -> np.ndarray[np.float32]:
        """Apply bandpass filter while maintaining state between chunks

        Args:
            signal (np.ndarray): Input signal (1D or 2D)

        Returns:
            np.ndarray[np.float32]: Filtered signal
        """

        if not self._initialized and signal.shape[0] > 0:
            for i in range(signal.shape[1]):
                self.zi[i] = self.zi[i] * signal[0, i]
            self._initialized = True

        filtered_signal = np.zeros_like(signal)
        for i in range(signal.shape[1]):
            filtered_signal[:, i], self.zi[i] = sosfilt(self.sos, signal[:, i], zi=self.zi[i])
        return filtered_signal.astype(np.float32)


class EmptyPipeline:
    @staticmethod
    def no_changes(signal: np.ndarray) -> np.ndarray:
        """Apply no changes to the input signal and return it as is

        Args:
            signal (np.ndarray): The input signal to be returned without any modifications
            np.ndarray: The input signal unchanged
        """
        return signal