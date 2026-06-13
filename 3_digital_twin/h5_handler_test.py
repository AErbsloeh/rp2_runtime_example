import unittest
from unittest.mock import patch, MagicMock
from numpy import float32
from h5_handler import H5Handler, H5Metadata

class H5HandlerTest(unittest.TestCase):
    def setUp(self):
        self.handler = H5Handler.__new__(H5Handler)  # Create an instance without calling __init__

        self.handler._recording_name = "test_recording"
        self.handler._metadata = H5Metadata(measurement_duration=12,
                                    sampling_rate=67,
                                    process_pipeline=["func1", "func2"],
                                    number_of_channels=8)
        self.handler._data_typ = float32
        self.handler._file_length = 0
        self.handler._num_of_data_in_buffer = 0


    @patch ("h5py.File")
    def test_init_h5_file_writer(self, mock_h5file):
        mock_file = MagicMock()
        mock_h5file.return_value = mock_file
        mock_group = MagicMock()
        mock_file.create_group.return_value = mock_group
        
        self.handler._init_h5_file_writer()
        mock_h5file.assert_called_with("test_recording.h5", "w")
        mock_group.attrs.__setitem__.assert_any_call("measurement_duration", 12)
        mock_group.attrs.__setitem__.assert_any_call("sampling_rate", 67)
        mock_group.attrs.__setitem__.assert_any_call("process_pipeline", "func1, func2")
        mock_group.attrs.__setitem__.assert_any_call("number_of_channels", 8)
        mock_group.create_dataset.assert_any_call('timestamps', shape=(0,), maxshape=(None,), dtype='float64', chunks=True)
        mock_group.create_dataset.assert_any_call('measurements', shape=(0, 8), maxshape=(None, 8), dtype=float32, chunks=True)
    

    def test_append_data(self):
        self.handler._h5file = MagicMock()
        self.handler._grp_ad7779 = {
            "timestamps": MagicMock(),
            "measurements": MagicMock(),
        }
        
        timestamps = [1001, 1002]
        measurements = [[10,20,30,40,50,60,70,80], [11,21,31,41,51,61,71,81]]
        
        self.handler.append_data(timestamps, measurements)

        self.handler._grp_ad7779["timestamps"].resize.assert_called_with((2,))
        self.handler._grp_ad7779["measurements"].resize.assert_called_with((2, 8))
