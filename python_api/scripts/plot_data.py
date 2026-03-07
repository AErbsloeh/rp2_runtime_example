import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from logging import basicConfig, DEBUG, INFO
from api import DataAPI, RawRecording, get_path_to_project


def plot_histogram_time(packet: RawRecording, show_density: bool=False, show_plot: bool=True) -> None:
    plt.figure()
    plt.hist(np.diff(packet.time[1:-1]), density=show_density, cumulative=show_density, histtype="stepfilled", color='k')
    plt.xlabel('Sampling Period (s)')
    plt.ylabel('Bins')
    plt.grid(True)
    plt.tight_layout()
    if show_plot:
        plt.show()
        
        
def plot_transient_time(packet: RawRecording, do_logy: bool=False, show_plot: bool=True) -> None:
    plt.figure()
    plt.plot(packet.time[1:-1], np.diff(packet.time[:-1]), linewidth=1, marker='.', markersize=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Sampling time')
    plt.yscale('log' if do_logy else 'linear')
    plt.xlim(packet.time[0], packet.time[-1])
    plt.grid(True)
    plt.tight_layout()
    if show_plot:
        plt.show()


def plot_transient_drift(packet: RawRecording, do_logy: bool=False, show_plot: bool=True) -> None:
    dt = np.diff(packet.time[:-1]) - 1/packet.sampling_rate

    drift_time = np.zeros_like(dt)
    for idx, dt0 in enumerate(dt, start=1):
        if idx == dt.size:
            break
        else:
            drift_time[idx] = dt0 + drift_time[idx-1]

    plt.figure()
    plt.plot(packet.time[1:-1], drift_time, linewidth=1, marker='.', markersize=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Drift time')
    plt.yscale('log' if do_logy else 'linear')
    plt.xlim(packet.time[0], packet.time[-1])
    plt.grid(True)
    plt.tight_layout()
    if show_plot:
        plt.show()


def plot_transient_data(packet: RawRecording, show_plot: bool=True) -> None:
    min_size = np.min([packet.time.size, packet.data.shape[1]])-1

    plt.figure()
    plt.plot(packet.time[:min_size], packet.data[:,:min_size].T, linewidth=1, marker='.', markersize=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.xlim(packet.time[0], packet.time[-1])
    plt.grid(True)
    plt.tight_layout()
    if show_plot:
        plt.show()


def plot_transient_util(packet: RawRecording, show_plot: bool=True) -> None:
    min_size = np.min([packet.time.size, packet.data.shape[1]])-1

    plt.figure()
    plt.plot(packet.time[:min_size], packet.data[0,:min_size], marker='.', color='k', label='CPU')
    plt.plot(packet.time[:min_size], packet.data[1,:min_size], marker='.', color='r', label='RAM')
    plt.xlabel('Time (s)')
    plt.ylabel('Utilization (%)')
    plt.legend(loc='best')
    plt.xlim(packet.time[0], packet.time[-1])
    plt.grid(True)
    plt.tight_layout()
    if show_plot:
        plt.show()


if __name__ == "__main__":
    basicConfig(level=INFO)
    read_util = False

    path2data = Path(get_path_to_project()) / "temp_data"
    use_case = -1

    dut = DataAPI(path2data, data_prefix='data')
    data = dut.read_data_file(use_case)

    if read_util:
        util = dut.read_utilization_file(use_case)
        plot_transient_util(util, show_plot=False)

    plot_transient_data(data, show_plot=False)
    plot_histogram_time(data, show_plot=False)
    plot_transient_time(data, do_logy=False, show_plot=False)
    plot_transient_drift(data, do_logy=False, show_plot=True)
