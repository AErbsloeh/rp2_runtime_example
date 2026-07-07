import matplotlib.pyplot as plt
import numpy as np
from shutil import rmtree
from api import get_path_to_project, DataAPI, DeviceAPI
from pathlib import Path
from time import sleep
from logging import basicConfig, DEBUG, INFO


def run_experiment(sampling_rate: float, do_batch: bool, run_trial_sec: float, repeat_trial: int=4, folder_name: str="bench_data") -> None:
    for _ in range(repeat_trial):
        DeviceAPI().do_reset()
        dut = DeviceAPI()
        dut.start_daq(
            sampling_rate=sampling_rate,
            do_batch=do_batch,
            do_plot=False,
            window_sec=10.,
            folder_name=folder_name
        )
        dut.wait_daq(run_trial_sec)
        dut.stop_daq()
        dut.close()
        sleep(1)


def extract_results(path: Path) -> tuple[list, list]:
    freq = list()
    results = list()

    dut = DataAPI(path)
    for idx, file in enumerate(path.glob("*_filt.h5")):
        dut.select_file(idx)
        data = dut.get_data('filt')
        dt = np.diff(data.time)
        results.append(dt)
        freq.append(data.sampling_rate)
    return freq, results


def plot_results(sampling_rate: np.ndarray, samp_rate: list, dt_data: list) -> None:
    efficiency = np.zeros(shape=(len(dt_data), ))
    period_minmax = np.zeros_like(efficiency)
    for idx, (data, srate) in enumerate(zip(dt_data, samp_rate)):
        efficiency[idx] = 1 / (np.median(data) * srate)
        period_minmax[idx] = np.std(data * srate)

    plt.figure()
    plt.plot(sampling_rate, np.ones_like(sampling_rate), color='r', linewidth=1.5, marker='.', markersize=4, label="Ideal")
    plt.errorbar(samp_rate, efficiency, period_minmax, color='k', linewidth=1.5, marker='.', markersize=4, label="Real")

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xscale('log')
    plt.xlim([0.99*samp_rate[0], 1.01*samp_rate[-1]])
    plt.xlabel('Sampling Rate (Hz)', fontsize=14)
    plt.ylabel('Coefficient', fontsize=14)
    plt.legend(loc='best', fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    basicConfig(level=INFO)
    # --- Experiment definition
    path2data = Path(get_path_to_project()) / "bench_data_batch"
    sampling_rate_range = np.logspace(start=1, stop=4, num=15, endpoint=False)
    num_repetitions = 1
    only_extract = True

    # --- Experiment run
    if not only_extract:
        if path2data.exists():
            rmtree(path2data, ignore_errors=True)

        for fs in sampling_rate_range:
            run_test_sec = 10
            print(f"Run trial with Sampling rate: {fs} Hz")
            run_experiment(
                sampling_rate=fs,
                do_batch=True,
                run_trial_sec=run_test_sec,
                repeat_trial=num_repetitions,
                folder_name=path2data.name
            )

    # --- Evaluation
    samp_rat, results = extract_results(path2data)
    plot_results(sampling_rate_range, samp_rat, results)
