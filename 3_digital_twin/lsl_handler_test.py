import unittest
from unittest.mock import patch, MagicMock
from lsl_handler import LSLHandler
from numpy import float32
from pylsl import cf_float32

class TestLSLHandler(unittest.TestCase):
    def setUp(self):
        self.lsl_handler = LSLHandler().__new__(LSLHandler)
    
    @patch("lsl_handler.StreamInfo")
    @patch("lsl_handler.StreamOutlet")
    def test_create_lsl_outlet(self, mock_stream_outlet, mock_stream_info):
        outlet = self.lsl_handler.create_lsl_outlet(name="TestStream", 
                                                    number_of_channels=4, 
                                                    channel_format=float32, 
                                                    sampling_rate=1000.0, 
                                                    type="data")
        
        mock_stream_info.assert_called_once_with(name="TestStream",
                                                 type="data",
                                                 channel_count=4,
                                                 nominal_srate=1000.0,
                                                 channel_format=cf_float32,
                                                 source_id="TestStream_uid")
        
        mock_stream_outlet.assert_called_once_with(mock_stream_info.return_value)
        self.assertEqual(outlet, mock_stream_outlet.return_value)

    @patch("lsl_handler.resolve_bypred")
    @patch("lsl_handler.StreamInlet")
    def test_connect_to_lsl_stream_success(self, mock_stream_inlet, mock_resolve):
        mock_resolve.return_value = ["Stream1"]
        inlet = self.lsl_handler.connect_to_lsl_stream(lsl_layer_name="Stream1")
        
        mock_resolve.assert_called_once_with(predicate="name='Stream1'")
        mock_stream_inlet.assert_called_once_with(
            "Stream1",
            max_buflen=60,
            max_chunklen=1024,
            recover=True,
            processing_flags=8
        )
        self.assertEqual(inlet, mock_stream_inlet.return_value)

    
    @patch("lsl_handler.resolve_bypred")
    @patch("lsl_handler.StreamInlet")
    def test_connect_to_lsl_stream_failure(self, mock_stream_inlet, mock_resolve):
        mock_resolve.return_value = []
        
        with  self.assertRaises(RuntimeError) as context:
            self.lsl_handler.connect_to_lsl_stream(lsl_layer_name="Stream2")
