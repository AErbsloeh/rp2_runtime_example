from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import get_args, get_origin, get_type_hints

import numpy as np
from pylsl import resolve_bypred

from api.src._helper import get_path_to_project
from api.src._live_visualize import build_comparison_plot_config
from api.src._lsl import ThreadLSL, numpy_to_lsl_format

TWIN_OUTPUT_STREAM = "DigitalTwinOutput"


@dataclass
class DigitalTwinConfig:
    """Configuration dataclass for the DigitalTwin
    Args:
        pipeline (list[Callable]): List of processing functions to be applied to the incoming data. Each function gets the output of the previous one. It is important that the output of the last function in the pipeline is a NumPy array, and that this function is annotated with a return type (e.g. -> np.ndarray[np.float32]) for the DigitalTwin to determine the output data type. The number of channels must not be changed by the pipeline
        lsl_layer_name_datasource (str): Name of the LSL stream to connect to for receiving data
        output_stream_name (str): Name of the LSL stream with the processed data
        save_h5 (bool): Flag indicating whether to record the processed stream into an H5 file (same format as the recordings of the DeviceAPI)
        path2save (Path | str | None): Folder for the H5 file, default is the folder "runs" of the project
        num_samples (int): Number of samples of one processing step (chunk size), a smaller value reduces the latency
    """

    pipeline: list[Callable]
    lsl_layer_name_datasource: str
    output_stream_name: str = TWIN_OUTPUT_STREAM
    save_h5: bool = True
    path2save: Path | str | None = None
    num_samples: int = 16


class DigitalTwin:
    _config: DigitalTwinConfig  # Configuration for the digital twin
    _threads: ThreadLSL  # Handler for the LSL threads (processing and recording), the same as used by the DeviceAPI
    _output_dtype: type  # Data type of the processed data, extracted from the annotation of the last pipeline function
    _is_running: bool  # Flag to indicate whether the digital twin is currently running

    def __init__(self, config: DigitalTwinConfig) -> None:
        """Loading configuration and preparing the digital twin. The LSL streams are connected during start()

        Args:
            config (DigitalTwinConfig): Configuration object containing the necessary settings for the digital twin

        Raises:
            ValueError: If the pipeline is empty
            TypeError: If the last function of the pipeline is not annotated with a NumPy array as return type
        """
        if not config.pipeline:
            raise ValueError("The pipeline needs at least one function")
        self._config = config
        self._output_dtype = self._format_output_pipeline(config.pipeline[-1])
        self._threads = ThreadLSL()
        self._is_running = False

    @property
    def source_stream_name(self) -> str:
        """Name of the LSL stream the digital twin receives its data from"""
        return self._config.lsl_layer_name_datasource

    @property
    def output_stream_name(self) -> str:
        """Name of the LSL stream with the processed data"""
        return self._config.output_stream_name

    def start(self) -> None:
        """Start the digital twin by launching the processing thread and (optional) the recording thread"""
        if self._is_running:
            return
        self._threads.register(
            func=self._threads.lsl_process_stream,
            args=(
                0,
                self.source_stream_name,
                self.output_stream_name,
                self._config.num_samples,
                lambda sampling_rate: None,
                self._process_chunk,
            ),
            kwargs={"lsl_format": numpy_to_lsl_format(self._output_dtype), "require_consumers": False},
        )
        if self._config.save_h5:
            path2save = self._config.path2save
            self._threads.register(
                func=self._threads.lsl_record_stream,
                args=(
                    1,
                    [self.output_stream_name],
                    get_path_to_project("runs") if path2save is None else path2save,
                ),
                kwargs={
                    "extra_attrs": {
                        "process_pipeline": ", ".join(map(self._get_function_name, self._config.pipeline))
                    }
                },
            )
        self._threads.start()
        self._is_running = True

    def wait(self, time_sec: float) -> None:
        """Waiting while the digital twin is running and raising errors of the threads (e.g. source stream not found)

        Args:
            time_sec (float): Time to wait in seconds
        """
        self._threads.wait_for_seconds(time_sec)

    def process(self, data: np.ndarray) -> np.ndarray:
        """Processing a whole array with the pipeline without LSL (offline), chunk by chunk with `num_samples` like the running twin does.
        The functions of the pipeline keep their state (e.g. filters), so call it once per run or create a new twin

        Args:
            data (np.ndarray): Samples as [samples] (one channel) or [samples, channels]

        Raises:
            ValueError: If the array has no samples

        Returns:
            np.ndarray: Processed samples as [samples, channels] with the output data type of the pipeline
        """
        values = np.asarray(data)
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        if len(values) == 0:
            raise ValueError("No samples to process")
        size = self._config.num_samples
        chunks = [
            np.asarray(self._process_chunk(values[start : start + size].tolist()))
            for start in range(0, len(values), size)
        ]
        return np.concatenate(chunks).astype(self._output_dtype)

    def plot(
        self,
        window_sec: float = 10.0,
        source_unit: str = "LSB",
        twin_unit: str = "LSB",
        extra_streams: list[tuple[str, str, str]] | None = None,
        error_streams: list[tuple[str, str, str]] | None = None,
        title_info: Callable[[], str] | None = None,
    ) -> None:
        """Open the live plot with the source stream in the first row and the processed stream after the extra streams and the error streams in the last rows (one column per channel). This call blocks until the plot window is closed

        Args:
            window_sec (float, optional): Window length in seconds. Defaults to 10.0
            source_unit (str, optional): Unit of the source data. Defaults to "LSB"
            twin_unit (str, optional): Unit of the processed data. Defaults to "LSB"
            extra_streams (list, optional): Further LSL streams with the same channels (e.g. data returned from the device) as rows between source and twin, each entry is (LSL stream name, label of the row, unit). Defaults to None
            title_info (Callable, optional): Function with a text for the title of the window next to the runtime (e.g. the running MAE of the `ErrorMonitor`). Defaults to None
            error_streams (list, optional): Further LSL streams with the same channels (e.g. the error between device and twin from the `ErrorMonitor`) as rows below the twin, each entry is (LSL stream name, label of the row, unit). Defaults to None
        """
        streams = resolve_bypred(predicate=f"name='{self.source_stream_name}'", minimum=1, timeout=10.0)
        if not streams:
            raise ValueError(f"LSL stream with {self.source_stream_name} not found")
        plot_config = build_comparison_plot_config(
            rows=[
                (self.source_stream_name, "Input", source_unit),
                *(extra_streams or []),
                (self.output_stream_name, "Twin", twin_unit),
                *(error_streams or []),
            ],
            num_channels=streams[0].channel_count(),
        )
        self._threads.lsl_plot_stream(
            plot_config=plot_config, window_length=window_sec, title_info=title_info
        )

    def stop(self) -> None:
        """Stop the digital twin by signaling the threads to terminate and waiting for them to finish"""
        if not self._is_running:
            return
        self._threads.stop()
        self._is_running = False

    def _process_chunk(self, chunk: list) -> list:
        """Apply the pipeline on one chunk of the source stream

        Args:
            chunk (list): Chunk with the data of the source stream, [samples][channels]

        Returns:
            list: Processed chunk, [samples][channels]
        """
        data = np.asarray(chunk)
        for func in self._config.pipeline:
            data = func(data)
        return np.asarray(data).tolist()

    @staticmethod
    def _get_function_name(func: Callable) -> str:
        if isinstance(func, partial):
            func = func.func
        return getattr(func, "__name__", str(func))

    @staticmethod
    def _format_output_pipeline(pipeline_function: Callable) -> type:
        """Determine the output data type of the given function

        Args:
            pipeline_function (callable): Function for which to determine the ouput data type

        Raises:
            TypeError: If the output type of the function is not a NumPy array or is not properly annotated with a return type

        Returns:
            type: The output data type of the function, extracted from its return type annotation
        """
        return_type = get_type_hints(pipeline_function).get("return")

        if get_origin(return_type) is not np.ndarray:
            raise TypeError(
                "The output of the last function in the pipeline must be a NumPy array, and it must be annotated with the return type. For example: def func(...) -> np.ndarray[np.float32]: ..."
            )

        args = get_args(return_type)
        if args:
            return args[0]
        return return_type
