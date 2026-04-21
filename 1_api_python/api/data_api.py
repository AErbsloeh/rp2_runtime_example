import h5py
import numpy as np
from dataclasses import dataclass
from logging import getLogger, Logger
from pathlib import Path


@dataclass(frozen=True)
class RawRecording:
    """Data class with measured transient data
    Attributes:
        sampling_rate:  Float with defined sampling rate [Hz]
        num_channels:   Integer with number of channels
        time:           Numpy array with timestamps [sec]
        data:           Numpy array with raw data
        file:           String with path to file
        type:           Type of data, e.g. "sensor" or "utilization"
    """
    sampling_rate: float
    num_channels: int
    time: np.ndarray
    data: np.ndarray
    type: str
    file: str

    @property
    def get_sampling_rate(self) -> float:
        return float(1/np.mean(np.diff(self.time)))

    @property
    def get_period_std(self) -> float:
        return float(np.std(np.diff(self.time)))


class DataAPI:
    __overview: list[Path]
    __logger: Logger
    __file: Path

    def __init__(self, path2data: Path | str, data_prefix: str="data") -> None:
        """Class for loading and processing the measured DAQ data
        :param path2data:   Path or string with path to the folder in which data is saved
        :param data_prefix: String with prefix of the data file name
        :return:            None
        """
        path = Path(path2data) if type(path2data) == str else path2data
        self.__logger = getLogger(__name__)
        self.__overview = [file.absolute() for file in path.glob(f"*{data_prefix}.h5")]

    @property
    def get_overview_files(self) -> list[Path]:
        """Returning a list with data files in the folder"""
        return [file for file in self.__overview]

    @property
    def is_utilization_available(self) -> bool:
        """Returning True if utilization data is available, False otherwise"""
        try:
            self.get_utilization()
        except ValueError:
            return False
        else:
            return True

    def select_file(self, file_number: int) -> None:
        """Selecting file by number
        :param file_number: Integer with number of the file
        :return:            None
        """
        self.__file = self.get_overview_files[file_number]
        self.__logger.info(f"Read file: {self.__file}")

    def get_file_name(self) -> str:
        """Returning the data file name of the corresponding use case"""
        return str(self.__file.name)

    def get_path2file(self) -> str:
        """Returning the path to the data file"""
        return str(self.__file)

    def __read_data_file(self, path2file: Path) -> RawRecording:
        with h5py.File(path2file, "r") as f:
            self.__logger.info(f"Datasets in file: {list(f.keys())}")
            self.__logger.info(f"Meta info: {list(f.attrs.keys())}")
            data = RawRecording(
                sampling_rate=f.attrs["sampling_rate"],
                num_channels=f.attrs["channel_count"],
                time=np.array(f["data_time"][:] - f["data_time"][0]),
                data=np.transpose(f["data_raw"][:]),
                type=f.attrs["type"],
                file=str(path2file)
            )
            f.close()
        return data

    def get_sensor_data(self) -> RawRecording:
        """Reading data content from file
        :return:                Class RawRecording with information, timestamps and raw data
        """
        if not self.__file:
            raise ValueError("Please select a file first!")
        return self.__read_data_file(self.__file)

    def __read_util_file(self, path2file: Path) -> RawRecording:
        with h5py.File(path2file, "r") as f:
            self.__logger.info(f"Datasets in file: {list(f.keys())}")
            self.__logger.info(f"Meta info: {list(f.attrs.keys())}")
            if not f.attrs["is_util_tracked"]:
                raise ValueError("Utilization data is not available - Please disable the option!")
            else:
                time = np.array(f["util_time"][:] - f["util_time"][0])
                sampling_rate = float(1 / np.mean(np.diff(time)))
                data = RawRecording(
                    sampling_rate=sampling_rate,
                    num_channels=2,
                    time=time,
                    data=np.transpose(f["util_data"][:]),
                    type="utilization",
                    file=str(path2file)
                )
                f.close()
                return data

    def get_utilization(self) -> RawRecording:
        """Reading utilization content from file
        :return:                Class RawRecording with information, timestamps and raw data
        """
        if not self.__file:
            raise ValueError("Please select a file first!")
        return self.__read_util_file(self.__file)
