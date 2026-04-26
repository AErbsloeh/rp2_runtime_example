import h5py
import numpy as np
from dataclasses import dataclass
from logging import getLogger, Logger
from pathlib import Path


@dataclass(frozen=True)
class StreamRecording:
    """Data class with recorded transient data
    Attributes:
        sampling_rate:  Float with defined sampling rate [Hz]
        num_channels:   Integer with number of channels
        time:           Numpy array with timestamps [sec]
        data:           Numpy array with raw data
        file:           String with path to file
        type:           Type of data, e.g. "sensor" or "utilization"
        units:          List with string of unit information, like "%", "V"
        label:          List with string name / label of each channel
        layout:         Layout information with where is each electrode device
    """
    sampling_rate: float
    num_channels: int
    time: np.ndarray
    data: np.ndarray
    type: str
    file: str
    units: list[str]
    label: list[str]
    layout: list[int]

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

    def __init__(self, path2data: Path | str) -> None:
        """Class for loading and processing the measured DAQ data
        :param path2data:   Path or string with path to the folder in which data is saved
        :return:            None
        """
        path = Path(path2data) if type(path2data) == str else path2data
        self.__logger = getLogger(__name__)
        self.__overview = [file.absolute() for file in path.glob("*.h5")]

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

    def get_overview_datatypes(self) -> list[str]:
        """Get a list with available datatypes to load from file"""
        if not self.__file:
            raise ValueError("Please select a file first!")
        with h5py.File(self.__file, "r") as f:
            overview = [type.split("_")[0] for type in f.keys() if "_data" in type]
            f.close()
        return overview

    def select_file(self, file_number: int) -> None:
        """Selecting file by number for processing
        :param file_number: Integer with number of the file
        :return:            None
        """
        if not self.get_overview_files:
            raise ValueError(f"No files available at : {self.__overview}")
        if file_number > len(self.get_overview_files):
            raise ValueError(f"Only {len(self.get_overview_files)} files are available")
        self.__file = self.get_overview_files[file_number]
        self.__logger.info(f"Read file: {self.__file}")

    def get_file_name(self) -> str:
        """Returning the data file name of the corresponding use case"""
        return str(self.__file.name)

    def get_path2file(self) -> str:
        """Returning the path to the data file"""
        return str(self.__file)

    def __read_data_file(self, name: str) -> StreamRecording:
        with h5py.File(self.__file, "r") as f:
            self.__logger.info(f"Datasets in file: {list(f.keys())}")
            self.__logger.info(f"Meta info: {list(f.attrs.keys())}")

            data = StreamRecording(
                sampling_rate=f[f"{name}_data"].attrs["sampling_rate"],
                num_channels=f[f"{name}_data"].attrs["channel_count"],
                time=np.array(f[f"{name}_time"][:] - f[f"{name}_time"][0]),
                data=np.transpose(f[f"{name}_data"][:]),
                type=f[f"{name}_data"].attrs["type"],
                file=str(self.__file),
                units=f[f"{name}_data"].attrs["unit"],
                label=f[f"{name}_data"].attrs["label"],
                layout=f[f"{name}_data"].attrs["layout"],
            )
            f.close()
        return data

    def get_data(self, name: str) -> StreamRecording:
        """Reading data content from file
        :param name:    Name of the dataset subset, available in the file
        :return:        Class RawRecording with information, timestamps and raw data
        """
        if not self.__file:
            raise ValueError("Please select a file first!")
        if not name in self.get_overview_datatypes():
            raise ValueError(f"Dataset with {name} is not available ({self.get_overview_datatypes()}")
        return self.__read_data_file(name)

    def get_utilization(self) -> StreamRecording:
        """Reading utilization content from file
        :return:        Class RawRecording with information, timestamps and raw data
        """
        return self.get_data('util')
