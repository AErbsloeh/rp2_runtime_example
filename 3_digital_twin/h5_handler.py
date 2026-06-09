import h5py
import time
from dataclasses import dataclass
from numpy import dtype

@dataclass
class H5Metadata:
    measurement_duration: int
    sampling_rate: int
    process_pipeline: list[str]
    number_of_channels: int


class H5Handler:
    _recording_name: str
    _metadata: H5Metadata
    _h5file: h5py.File
    _grp_ad7779: h5py.Group
    _length_ad7779: int

    def __init__(self, recording_name: str, metadata: H5Metadata, data_typ= dtype) -> None:
        """Class to handle H5 file writing for EEG data, including initialization and appending data

        Args:
            recording_name (str): Filename for the H5 file
            metadata (H5Metadata): Metadata information about the measurement
            data_typ (dtype): The data type for the H5 dataset
        """  
        self._recording_name = recording_name
        self._metadata = metadata
        self._data_typ = data_typ
        self._h5file, self._grp_ad7779 = self._init_h5_file_writer()
        self._num_of_data_in_buffer = 0
        self._file_length = 0

    @property
    def get_file_length(self) -> int:
        """Get the current length of the ad7779 dataset in the H5 file

        Returns:
            int: The length of the ad7779 dataset
        """        
        return self._file_length


    def _init_h5_file_writer(self) -> tuple[h5py.File, h5py.Group]:
        """Initialize the H5 file and create necessary groups and datasets

        Returns:
            tuple[h5py.File, h5py.Group]: The H5 file object and the group for ad7779 data
        """
        file =h5py.File(f"{self._recording_name}.h5", "w")
        # Output Metadata as attributes
        file.attrs['created_at'] = time.ctime()
        file.attrs["version"] = "1.0"

        # generate group for ad7779
        grp_digital_twin = file.create_group("digital_twin")
        
        #Write metadata attributes
        grp_digital_twin.attrs["measurement_duration"] = self._metadata.measurement_duration
        grp_digital_twin.attrs["sampling_rate"] = self._metadata.sampling_rate
        grp_digital_twin.attrs["process_pipeline"] = ', '.join(self._metadata.process_pipeline)
        grp_digital_twin.attrs["number_of_channels"] = self._metadata.number_of_channels

        # Create datasets with maxshape for appending data
        grp_digital_twin.create_dataset('timestamps', shape=(0,), maxshape=(None,), dtype='float64', chunks=True)
        grp_digital_twin.create_dataset('measurements', shape=(0, self._metadata.number_of_channels), maxshape=(None, self._metadata.number_of_channels), dtype=self._data_typ, chunks=True)
        return file, grp_digital_twin


    def append_data(self, timestamps: float, measurements: list) -> None:
        """Append data to the ad7779 datasets in the H5 file

        Args:
            timestamps (int): timestamp value to the datapoint
            measurements (list): list of measurement values for all channels at the data point
            alerts (list): a list of alert bits for all channels of the data point
        """
        dset_time =self._grp_ad7779["timestamps"]
        dset_meas =self._grp_ad7779["measurements"]

        current_length = self._file_length
        self._file_length += len(timestamps)

        dset_time.resize((self._file_length,))
        dset_meas.resize((self._file_length, self._metadata.number_of_channels))
        
        dset_time[current_length:self._file_length] = timestamps
        dset_meas[current_length:self._file_length, :] = measurements
        
        if self._num_of_data_in_buffer >= 10:
            self._h5file.flush()
            self._num_of_data_in_buffer = 0
        else:
            self._num_of_data_in_buffer += 1
            

    def close_h5_file(self) -> None:
        """Close the H5 file properly"""
        self._h5file.flush()
        self._h5file.close()