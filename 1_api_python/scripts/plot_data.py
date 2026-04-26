import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from logging import basicConfig, DEBUG, INFO
from api import DataAPI, StreamRecording, get_path_to_project


def plot_histogram_time(packet: StreamRecording, show_density: bool=False, show_plot: bool = True) -> None:
    data = 1e3 * np.diff(packet.time[1:-1])
    num_bins = np.unique(data).size

    plt.figure()
    plt.hist(data, bins=num_bins, align="mid", cumulative=show_density, histtype="stepfilled", color='k')
    plt.xlabel('Sampling Period (ms)')
    plt.ylabel('Bins')
    plt.grid(True)
    plt.tight_layout()
    if show_plot:
        plt.show()


def plot_transient_time(packet: StreamRecording, do_logy: bool = False, show_plot: bool = True) -> None:
    plt.figure()
    plt.plot(packet.time[:-1], 1e3 * np.diff(packet.time), linewidth=1, marker='.', markersize=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Sampling time (ms)')
    plt.yscale('log' if do_logy else 'linear')
    plt.xlim(packet.time[0], packet.time[-2])
    plt.grid(True)
    plt.tight_layout()
    if show_plot:
        plt.show()


def plot_transient_drift(packet: StreamRecording, do_logy: bool = False, show_plot: bool = True) -> None:
    dt = np.diff(packet.time) - 1 / packet.sampling_rate

    drift_time = np.zeros_like(dt)
    for idx, dt0 in enumerate(dt, start=1):
        if idx == dt.size:
            break
        else:
            drift_time[idx] = dt0 + drift_time[idx - 1]

    plt.figure()
    plt.plot(packet.time[:-1], 1e6 * drift_time, linewidth=1, marker='.', markersize=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Drift time (µs)')
    plt.yscale('log' if do_logy else 'linear')
    plt.xlim(packet.time[0], packet.time[-2])
    plt.grid(True)
    plt.tight_layout()
    if show_plot:
        plt.show()


def plot_transient_data(packet: StreamRecording, show_plot: bool = True) -> None:

    plt.figure()
    plt.plot(packet.time, packet.data.T, linewidth=1.5, marker='.', markersize=2)
    plt.xlabel('Time (s)')
    plt.ylabel(f'Value ({packet.units[0]})')
    plt.xlim(packet.time[0], packet.time[-1])
    plt.legend(packet.label, loc='best')
    plt.grid(True)
    plt.tight_layout()
    if show_plot:
        plt.show()


def plot_transient_util(packet: StreamRecording, show_plot: bool = True) -> None:
    plt.figure()
    plt.plot(packet.time, packet.data[0, :], marker='.', color='k', label='CPU')
    plt.plot(packet.time, packet.data[1, :], marker='.', color='r', label='RAM')
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

    dut = DataAPI(path2data)
    dut.select_file(use_case)
    data = dut.get_data('mock')

    if read_util and dut.is_utilization_available:
        util = dut.get_utilization()
        plot_transient_util(util, show_plot=False)

    plot_transient_data(data, show_plot=False)
    plot_histogram_time(data, show_plot=False)
    plot_transient_time(data, do_logy=False, show_plot=False)
    plot_transient_drift(data, do_logy=False, show_plot=True)
