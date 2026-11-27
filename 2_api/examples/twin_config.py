import numpy as np


class EmptyPipeline:
    @staticmethod
    def no_changes(signal: np.ndarray) -> np.ndarray:
        """Apply no changes to the input signal and return it as is

        Args:
            signal (np.ndarray): The input signal to be returned without any modifications
            np.ndarray: The input signal unchanged
        """
        return signal


class FirmwareProcessing:
    @staticmethod
    def run_sequential(signal: np.ndarray, in_bytes_value: int = 2, out_bytes_value: int = 2) -> np.ndarray[np.int32]:
        """Model of `run_sequential` in `0_firmware/src/user_logic.c` for the first channel of the input block:
        Each byte of the value is shifted one bit to the right (value >> 1 per byte, not for the whole value).
        The value is cut to the bytes of one input value (two's complement, like `DeviceAPI.process_data`), the result has the bytes of one output value (unsigned).
        Change this function together with the firmware.

        Args:
            signal (np.ndarray): The input values (integer, any shape), e.g. the DAC codes sent to the device
            in_bytes_value (int, optional): Number of bytes of one value of the input block (`processing_config.in.element_size`). Defaults to 2.
            out_bytes_value (int, optional): Number of bytes of one value of the output block (`processing_config.out.element_size`). Defaults to 2.

        Returns:
            np.ndarray[np.int32]: The values processed like the firmware does, same shape as the input
        """
        raw = np.asarray(signal).astype(np.int64) & ((1 << (8 * in_bytes_value)) - 1)
        result = np.zeros_like(raw)
        for idx in range(min(in_bytes_value, out_bytes_value)):
            result |= (((raw >> (8 * idx)) & 0xFF) >> 1) << (8 * idx)
        return result.astype(np.int32)
