import datetime
from collections.abc import Callable
from dataclasses import dataclass
from typing import get_type_hints, get_origin, get_args
from pylsl import StreamInlet, StreamOutlet
from threading import Thread
import numpy as np
from h5_handler import H5Handler, H5Metadata
from lsl_handler import LSLHandler



@dataclass
class DigitalTwinConfig:
    """Configuration dataclass for the DigitalTwin
    Args:
        recording_name (str): Name for the recording, used for naming the H5 file
        pipeline (list[Callable]): List of processing functions to be applied to the incoming data. It is important that the output of the last function in the pipeline is a NumPy array, and that all functions are properly annotated with return types for the DigitalTwin to determine the output data type
        lsl_layer_name_datasource (str): Name of the LSL stream layer to connect to for receiving data
        measurement_duration (int): Duration of the measurement in seconds
        save_h5 (bool): Flag indicating whether to save the processed data in an H5 file
    """    
    recording_name: str
    pipeline: list[Callable]
    lsl_layer_name_datasource: str
    measurement_duration: int
    save_h5: bool


class DigitalTwin:
    _config: DigitalTwinConfig          # Configuration for the digital twin, including recording name, processing pipeline, LSL layer name, measurement duration, and whether to save data in H5 format
    _is_running: bool                   # Flag to indicate whether the digital twin is currently running
    _threads: list[Thread]              # List of threads used for data processing and recording
    _reccording_name: str               # Name for the h5 file, generated based on the provided recording name and the current timestamp
    _pipeline: list[Callable]           # List of processing functions that will be applied to the incoming data in sequence
    _inlet: StreamInlet                 # LSL inlet for receiving data from the specified LSL layer
    _outlet: StreamOutlet               # LSL outlet for sending processed data to a new LSL stream named "DigitalTwinOutput"
    _processed_data_inlet: StreamInlet  # LSL inlet for receiving processed data from the "DigitalTwinOutput" stream, used for recording purposes
    _deployed_h5_handler: H5Handler     # H5Handler instance for managing the H5 file where processed data will be recorded, initialized if save_h5 is True in the configuration
    def __init__(self, config: DigitalTwinConfig):
        """Loading configuraion and setting up the digital twin, including connecting to the specified LSL stream for data input, creating an LSL outlet for processed data output, and initializing the H5 handler if saving to H5 is enabled in the configuration

        Args:
            config (DigitalTwinConfig): Configuration object containing the necessary settings for the digital twin
        """        
        self._config = config
        self._pipeline = config.pipeline
        self._is_running = False
        self._threads = []
        self._reccording_name = f"{self._config.recording_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_rec_digi_twin"
        
        self._inlet = LSLHandler.connect_to_lsl_stream(self._config.lsl_layer_name_datasource) #Connect to the specified LSL stream layer to receive data for processing        
        
        self._output_format = self._format_output_pipeline(self._pipeline[-1]) #Determine the output data type of the last function in the processing pipeline
        self._outlet = LSLHandler.create_lsl_outlet(name ="DigitalTwinOutput", 
                                                        number_of_channels=(self._inlet.info().channel_count()+1), 
                                                        channel_format= np.floating) # Needs too be floating point to be able to include timestamps as well, which are float64, in the output stream
        
        self._processed_data_inlet = LSLHandler.connect_to_lsl_stream(lsl_layer_name ="DigitalTwinOutput")
        
        if self._config.save_h5:
            h5_metadata = H5Metadata(measurement_duration=self._config.measurement_duration,
                                    sampling_rate=int(self._inlet.info().nominal_srate()),
                                    process_pipeline=[func.__name__ if hasattr(func, 'func') else getattr(func, '__name__', str(func)) for func in self._pipeline],
                                    number_of_channels=self._inlet.info().channel_count())
            self._deployed_h5_handler = H5Handler(recording_name=self._reccording_name,
                                             metadata=h5_metadata, data_typ= self._output_format)


    def start(self) -> None:
        """Start the digital twin by launching the data processing and recording threads"""        
        if self._is_running:
            return
        self._is_running = True
        self._inlet.pull_chunk(timeout=0.001)  # Pull chunk to clear data
        
        self._threads.append(Thread(target=self._data_processing_loop, daemon=True))
        if self._config.save_h5:
            self._threads.append(Thread(target=self._record_data_loop, daemon=True))
        for thread in self._threads:
            thread.start()


    def stop(self) -> None:
        """Stop the digital twin by signaling the threads to terminate and waiting for them to finish"""        
        self._is_running = False
        for thread in self._threads:
            thread.join()
        self._threads = []


    def _format_output_pipeline(self, pipeline_function: Callable) -> np.dtype:
        """Determine the output data type of the given function

        Args:
            pipeline_function (callable): Function for which to determine the ouput data type

        Raises:
            TypeError: If the output type of the function is not a NumPy array or is not properly annotated with a return type

        Returns:
            type: The output data type of the function, extracted from its return type annotation
        """        
        self._output_format = get_type_hints(pipeline_function)
        return_type = self._output_format.get("return")

        if not get_origin(return_type) is np.ndarray:
            raise TypeError("The output of the last function in the pipeline must be a NumPy array, and it must be annotated with the return type. For example: def func(...) -> np.ndarray[np.float32]: ...")
        
        args = get_args(return_type)
        if args:
            return args[0]
        return return_type


    def _data_processing_loop(self) -> None:
        """Process data from the LSL inlet and push it to the output LSL outlet"""
        while self._is_running:
            data, timestamps = self._inlet.pull_chunk(timeout=0.0001, max_samples=1024)
            if not data:
                continue
            data = np.array(data)
            for func in self._pipeline:
                data = func(data)
            data = np.hstack((data, np.array(timestamps)[:, np.newaxis]))
            self._outlet.push_chunk(data.tolist(), pushthrough=True)


    def _record_data_loop(self) -> None:
        """Pull processed data from the output LSL inlet and record it in the H5 file using the H5Handler"""        
        while self._is_running:
            data, timestamps =self._processed_data_inlet.pull_chunk(timeout=0.0001, max_samples=1024)
            if not data:
                continue
            timestamps = np.array(data)[:,-1]
            data = np.array(data)[:,:-1]
            self._deployed_h5_handler.append_data(timestamps=timestamps, measurements=data)
        self._deployed_h5_handler.close_h5_file()