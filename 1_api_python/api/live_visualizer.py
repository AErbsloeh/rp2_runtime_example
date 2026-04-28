import numpy as np
from dataclasses import dataclass
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
from pylsl import StreamInlet, resolve_bypred, proc_threadsafe, local_clock
from collections import deque

@dataclass
class LivePlotterChannelConfig:
    """Cofiguration dataclass for each channel to be visualized in the live plotter
    Attributes:
        name (str): Name of the channel that will be displayed in the plot legend
        visualized_channel (int): Index of the channel in the LSL stream to be visualized
        lsl_layer_name (str): Name of the LSL stream layer to connect to
        curve_color (str, optional): Color of the curve in the plot. Defaults to None.
        value_translation_func (callable, optional): Function to translate raw data values. Defaults to None.
    """
    name: str
    visualized_channel: int
    lsl_layer_name: str
    window_width_sec: float
    curve_color: str ='b'
    value_translation_func: callable =None


class LivePlotter:
    def __init__(self, config: LivePlotterChannelConfig):
        self._translation_func = [i.value_translation_func for i in config]
        self._inlet = self._search_lsl_stream_and_connect([i.lsl_layer_name for i in config])
        self._fs = self._get_stream_samplingrate() if self._get_stream_samplingrate() >0 else 250
        self._visualized_channel = [i.visualized_channel for i in config]

        self.max_samples = int(self._fs * config.window_width_sec) if type(config) == LivePlotterChannelConfig else int(self._fs * config[0].window_width_sec)

        self.data_buffers = [np.zeros(self.max_samples) for _ in config]
        self.time_buffers = [np.zeros(self.max_samples) for _ in config]
        self.write_pointers = [0 for _ in self._inlet]
        self.caluclate_counter = 0

        self._app, self._win, self._plot_item, self._curves, self._freq_labels =self._init_plot([i.curve_color for i in config], [i.name for i in config])
        self._timer = self._init_timer()


    def _search_lsl_stream_and_connect(self, lsl_layer_name: list) -> list[StreamInlet]:
        """Search for an LSL streams by name and connecting to them
        
        Args:    
            lsl_layer_name (list): A list of LSL stream layer names to search for

        Raises:
            RuntimeError: If no stream with the specified layer name is found

        Returns:
            list[StreamInlet]: A list of connected StreamInlet objects
        """        
        inlets = []
        print("Search for LSL Stream..")
        for layer_name in lsl_layer_name:
            streams = resolve_bypred(predicate=f"name='{layer_name}'")
            if streams:
                print(f"LSL Stream '{layer_name}' found, connecting...")
                inlet = StreamInlet(streams[0],
                                   max_buflen= 60,
                                   max_chunklen= 1024,
                                   recover=True,
                                   processing_flags=proc_threadsafe)
                inlets.append(inlet)
            else:
                raise RuntimeError(f"No Stream with Layer Name {layer_name} found!")
        return inlets


    def _get_stream_samplingrate(self) -> int:
        """ Getting the highest nominal sampling rate of the connected LSL streams

        Returns:
            int: Highest nominal sampling rate of the streams
        """
        fs =[]
        for inlet in self._inlet:
            fs.append(inlet.info().nominal_srate())
        return int(max(fs))
    

    def _init_plot(self, curves_color: list[str], curves_name: list[str]) -> tuple:
        """Initialize the PyQtGraph plot for live data visualization

        Returns:
            tuple: A tuple containing the QApplication, GraphicsLayoutWidget, PlotItem, and PlotDataItem
        """
        app = QtWidgets.QApplication([])
        win = pg.GraphicsLayoutWidget(show=True, title="LSL Live Plot")
        plot_item = win.addPlot(title="Live EEG Data")
        plot_item.setLabel("left", "Amplitude", units="Data Points")
        plot_item.setLabel("bottom", "Time", units="s")
        plot_item.showGrid(x=True, y=True)
        plot_item.addLegend()

        curves = []
        for idx,selected_data_buffer in enumerate(self.data_buffers):
            curves.append(plot_item.plot(pen= "y" if curves_color is None else curves_color[idx], name=curves_name[idx]))
        
        # Create frequency overlay using a ViewBox with TextItems
        freq_legend = pg.ViewBox()
        freq_legend.setBackgroundColor((50, 50, 50, 180))
        freq_legend.setFixedWidth(180)
        freq_legend.setFixedHeight(20 + len(self.data_buffers) * 22)
        freq_legend.setRange(xRange=(0, 1), yRange=(0, 1), padding=0)
        freq_legend.setMouseEnabled(x=False, y=False)
        freq_legend.setMenuEnabled(False)

        freq_labels = []
        for idx in range(len(self.data_buffers)):
            color = curves_color[idx] if curves_color else "y"
            label = pg.TextItem(text=f"{curves_name[idx]}: -- Hz", color=color, anchor=(0, 0))
            label.setPos(0.05, 0.95 - (idx + 1) * (0.9 / (len(self.data_buffers) + 1)))
            freq_legend.addItem(label)
            freq_labels.append(label)

        win.addItem(freq_legend, row=0, col=1)
        
        title_label = pg.TextItem(text="Peak Frequency", color="w", anchor=(0, 0))
        title_label.setPos(0.05, 0.95)
        freq_legend.addItem(title_label)
        
        return app, win, plot_item, curves, freq_labels
    

    def _init_timer(self) -> QtCore.QTimer:
        """Initialize the QTimer for periodic plot updates

        Returns:
            QtCore.QTimer: The initialized QTimer object
        """        
        timer = QtCore.QTimer()
        timer.timeout.connect(self._update)
        timer.start(20)
        return timer


    def _update(self, iterr_threshold_for_calulaction: int=10) -> None:
        """Update the plot with new data from the LSL stream

        Args:
            iterr_threshold_for_calulaction (int, optional): Threshold for the calculation. Defaults to 10.
        """        
        for idx ,selected_inlet in enumerate(self._inlet):
            data, timestamp = selected_inlet.pull_chunk(timeout=0.0, max_samples=self.max_samples)
            if not data:
                continue
            data = np.array(data)[:,self._visualized_channel[idx]]
            timestamp = np.array(timestamp)
            end_pointer = self.write_pointers[idx] + len(data)
            if self._translation_func[idx] is not None:
                data = self._translation_func[idx](data)

            if end_pointer <= self.max_samples:
                self.data_buffers[idx][self.write_pointers[idx]:end_pointer] = data
                self.time_buffers[idx][self.write_pointers[idx]:end_pointer] = np.array(timestamp)
            else:
                break_point = self.max_samples - self.write_pointers[idx]
                self.data_buffers[idx][self.write_pointers[idx]:] = data[:break_point]
                self.data_buffers[idx][:len(data) - break_point] = data[break_point:]
               
                self.time_buffers[idx][self.write_pointers[idx]:] = timestamp[:break_point]
                self.time_buffers[idx][:len(data) - break_point] = timestamp[break_point:]
            self.write_pointers[idx] = (self.write_pointers[idx] + len(data)) % self.max_samples

        time_now = local_clock()
        for idx, curve in enumerate(self._curves):
            plot_data = np.concatenate((self.data_buffers[idx][self.write_pointers[idx]:], self.data_buffers[idx][:self.write_pointers[idx]]))
            plot_time = np.concatenate((self.time_buffers[idx][self.write_pointers[idx]:], self.time_buffers[idx][:self.write_pointers[idx]]))
            valid_mask = plot_time >0 #Check for valid timestamps
            if np.any(valid_mask):
                curve.setData(plot_time[valid_mask] - time_now, plot_data[valid_mask])
                if np.sum(valid_mask) >= self.max_samples and self.caluclate_counter  >=iterr_threshold_for_calulaction:
                    self._caluclate_frequency(idx = idx, data= plot_data[valid_mask], time= plot_time[valid_mask])
        if self.caluclate_counter  >=iterr_threshold_for_calulaction:
            self.caluclate_counter =0
        else:
            self.caluclate_counter +=1


    def _caluclate_frequency(self, idx: int, data: np.ndarray, time: np.ndarray) -> None:
        """Calculate the peak frequency of the signal using FFT

        Args:
            idx (int): Index of the Curve
            data (np.ndarray): Data array for frequency calculation, after concatenation
            time (np.ndarray): Time array corresponding to the data, after concatenation
        """            
        fs = 1/np.mean(np.diff(time[-1024:]))
        fft_values = np.abs(np.fft.rfft(data[-1024:]))
        freqs = np.fft.rfftfreq(1024, d=1.0/fs)
        max_idx = np.argmax(fft_values)
        peak_freq = freqs[max_idx]
        self._freq_labels[idx].setText(f"{peak_freq:.2f} Hz")


    def start(self):
        """Start the live plotter"""        
        QtWidgets.QApplication.instance().exec_()


def start_live_plotter(config: list) -> None:
    """Start the live plotter with the given configuration"""
    plotter = LivePlotter(config=config)
    plotter.start()


def translation_func_dac(value_to_translate: np.array, v_ref: float =5.) -> list:
    value_to_translate = v_ref * ((value_to_translate / (2**15)) -1)
    return value_to_translate


def translation_func_adc(value_to_translate: int) -> list:
    value_to_translate = value_to_translate * 1.25 / 2 ** 23
    return value_to_translate


if __name__ == "__main__":
    config = [LivePlotterChannelConfig(visualized_channel=0,
                                       name="DAC Data",
                                       lsl_layer_name="PlayerData",
                                       curve_color="r",
                                       value_translation_func=translation_func_dac),
              LivePlotterChannelConfig(visualized_channel=0,
                                       name="Digital Twin",
                                       lsl_layer_name="DigitalTwinOutput",
                                       curve_color="g",
                                       value_translation_func=None)]
    config_DAC = [LivePlotterChannelConfig(visualized_channel=0,
                                       name="DAC Data",
                                       lsl_layer_name="PlayerData",
                                       curve_color="r",
                                       value_translation_func=translation_func_dac)]
    plotter = LivePlotter(config=config_DAC)
    plotter.start()