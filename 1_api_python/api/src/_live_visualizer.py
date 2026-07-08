from logging import config

import numpy as np
from dataclasses import dataclass, field
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
from pylsl import StreamInlet, resolve_bypred, proc_threadsafe, local_clock
from collections import deque

@dataclass
class LivePlotterChannelConfig: # Describes what should be plotted for each channel in the live plotter, stores information about the channel and how to visualize it
    """Cofiguration dataclass for each channel to be visualized in the live plotter
    Attributes:
        name (str): Name of the channel that will be displayed in the plot legend
        visualized_channel (int): Index of the channel in the LSL stream to be visualized
        lsl_layer_name (str): Name of the LSL stream layer to connect to
        curve_color (str, optional): Color of the curve in the plot. Defaults to None.
        value_translation_func (callable, optional): Function to translate raw data values. Defaults to None.
        enabled (bool, optional): Whether the channel is enabled for visualization. Defaults to True.
        grid_row (int, optional): Row position of the plot in the grid display mode. Defaults to None
        grid_col (int, optional): Column position of the plot in the grid display mode. Defaults to None
    """
    name: str
    visualized_channel: int
    lsl_layer_name: str
    window_width_sec: float
    curve_color: str ='b'
    value_translation_func: callable =None
    enabled: bool =True  
    grid_row: int =None
    grid_col: int =None

@dataclass
class PlotSessionConfig: # Wraps the full plot session:which display mode to use, how many rows and columns in the grid, and which channels to visualize
    """Configuration dataclass for the live plot session
    Attributes:
        display_mode (str):"filtered" (only enabled channels get a grid slot) or "grid" (all channels get a grid slot, disabled channels are empty)
        channels (list[LivePlotterChannelConfig]): List of channel configurations to be visualized
    """
    display_mode: str 
    channels: list[LivePlotterChannelConfig]= field(default_factory=list) # List of channel configurations to be visualized(whether they are enabled or not)
   

class LivePlotter: # actually connects data from LSL stream to the plot and updates it in real time
    def __init__(self, config: PlotSessionConfig):
        channels=config.channels # retrieve the list of channel configurations from the PlotSessionConfig object
        self._display_mode = config.display_mode # retrieve the display mode from the PlotSessionConfig object

        self._translation_func = [i.value_translation_func for i in channels]
        self._inlet = self._search_lsl_stream_and_connect([i.lsl_layer_name for i in channels])
        self._fs = self._get_stream_samplingrate() if self._get_stream_samplingrate() >0 else 250
        self._visualized_channel = [i.visualized_channel for i in channels]
        self._grid_positions = [(i.grid_row, i.grid_col) for i in channels] #store the grid positions for each channel, which will be used to determine where to place the plots in the grid layout

        self.max_samples = int(self._fs * channels[0].window_width_sec) # calculate the maximum number of samples to store in the circular buffer based on the sampling rate and window width specified in the first channel configuration

        self.data_buffers = [np.zeros(self.max_samples) for _ in channels]#creates one buffer foir each channel, initialized to zeros with a length of max_samples
        self.time_buffers = [np.zeros(self.max_samples) for _ in channels]
        self.write_pointers = [0 for _ in self._inlet] #Pointer to keep track of where to write new data in the circular buffer for each channel
        self.caluclate_counter = 0 #Counter to control how often the frequency calculation is performed (to reduce computational load)
        self._enabled = [i.enabled for i in channels] #for each channel, store whether it is enabled or not, so that the plotter can skip disabled channels during updates

        self._app, self._win, self._plot_items, self._curves, self._freq_labels, self._curve_source_indices =self._init_plot([i.curve_color for i in channels], [i.name for i in channels],[i.enabled for i in channels])
        self._timer = self._init_timer()

    def _make_plot_item(self, win: pg.GraphicsLayoutWidget, row: int, col: int, title: str):
      """Create and configure one empty plot box (axes, labels, grid) at a given grid position."""
      plot_item = win.addPlot(row=row, col=col, title=title)
      plot_item.setLabel("left", "Amplitude", units="Data Points")
      plot_item.setLabel("bottom", "Time", units="s")
      plot_item.showGrid(x=True, y=True)
      return plot_item

    def _make_curve(self, plot_item, curves_color: list[str], curves_name: list[str], idx: int):
      """Attach a legend, curve and frequency label to a plot box for an enabled channel."""
      plot_item.addLegend()
      color = "y" if curves_color is None else curves_color[idx]
      curve = plot_item.plot(pen=color, name=curves_name[idx])
      freq_label = pg.TextItem(text=f"{curves_name[idx]}: -- Hz", color=color, anchor=(0, 0))
      plot_item.addItem(freq_label)
      freq_label.setPos(0, 0)
      return curve, freq_label
        


    def _search_lsl_stream_and_connect(self, lsl_layer_name: list) -> list[StreamInlet]: #Search for the specified LSL stream ,connect to it and create an inlet for data retrieval
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
        for layer_name in lsl_layer_name: #loops thriugh every stream name specified in the config and tries to connect to it
            streams = resolve_bypred(predicate=f"name='{layer_name}'") #Searches for an LsL stream whosenames matches the specified layer name
            if streams: #If a stream is found, it creates an inlet to connect to the stream and retrieve data from it
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
        for inlet in self._inlet:#self.inlet contains all stream connections
            fs.append(inlet.info().nominal_srate())#inlet.info() gives access to the stream info, nominal_srate() returns the sampling rate of the stream,fs.append() adds the sampling rate to the list of sampling rates for all streams
        return int(max(fs))
    

    def _init_plot(self, curves_color: list[str], curves_name: list[str], enabled: list[bool]) -> tuple:
        """Initialize the PyQtGraph plot for live data visualization

        Returns:
            tuple: A tuple containing the QApplication, GraphicsLayoutWidget, PlotItem, and PlotDataItem
        """
        app = QtWidgets.QApplication([]) #starting a new QApplication, which is necessary for any PyQt application. It manages the GUI application's control flow and main settings.
        win = pg.GraphicsLayoutWidget(show=True, title="LSL Live Plot - Live EEG Data")#creates the actual window for the plot, with a title "LSL Live Plot"

        win.resize(1400, 700)#sets the size of the window to 1400x700 pixels
        plots_per_row = 4

        #these lists will store the objects created in the loop
        plot_items = []
        curves = []
        freq_labels = []
        curve_source_indices = [] # each curve corresponds to a specific channel, and this list keeps track of which channel each curve is associated with

        if self._display_mode == "grid":
         for idx in range(len(self.data_buffers)):
            row, col = self._grid_positions[idx]
            plot_item = self._make_plot_item(win, row, col, curves_name[idx])
            if enabled[idx]:
                curve, freq_label = self._make_curve(plot_item, curves_color, curves_name, idx)
            else:
              curve, freq_label = None, None
            plot_items.append(plot_item)
            curves.append(curve)
            freq_labels.append(freq_label)
            curve_source_indices.append(idx)

        else:
         plots_per_row = 4
         visible_indices = [idx for idx, is_enabled in enumerate(enabled) if is_enabled]
         for position, idx in enumerate(visible_indices):
            row, col = position // plots_per_row, position % plots_per_row
            plot_item = self._make_plot_item(win, row, col, curves_name[idx])
            curve, freq_label = self._make_curve(plot_item, curves_color, curves_name, idx)
            plot_items.append(plot_item)
            curves.append(curve)
            freq_labels.append(freq_label)
            curve_source_indices.append(idx)

        return app, win, plot_items, curves, freq_labels, curve_source_indices

        

    

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
            if not self._enabled[idx]: #If the channel is not enabled, skip the update for this channel and move to the next one
                continue 
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
        QtWidgets.QApplication.instance().exec()

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
