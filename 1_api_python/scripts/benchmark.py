import numpy as np
from shutil import rmtree
from api import get_path_to_project, DataAPI, DeviceAPI
from pathlib import Path
from time import sleep
from logging import basicConfig, DEBUG, INFO


def run_experiment(sampling_rate: float, do_batch: bool, run_trial_sec: float, repeat_trial: int=4, folder_name: str="bench_data") -> None:
    """"""
    for _ in range(repeat_trial):
        DeviceAPI().do_reset()
        dut = DeviceAPI()
        dut.start_daq(
            sampling_rate=sampling_rate,
            do_batch=do_batch,
            do_plot=False,
            window_sec=10.,
            track_util=False,
            folder_name=folder_name
        )
        dut.wait_daq(run_trial_sec)
        dut.stop_daq()
        dut.close()
        sleep(1)


def extract_results(path: Path) -> list:
    results = list()

    dut = DataAPI(path, data_prefix='data')
    for idx, file in enumerate(path.glob("*_data.h5")):
        data = dut.get_sensor_data(idx)
        dt = np.diff(data.time)
        results.append(dt)
    return results


def plot_results(dt_data: list) -> None:
    """"""
    pass


if __name__ == "__main__":
    sampling_rate_range = np.logspace(start=1, stop=4, num=10, endpoint=False)
    num_repetitions = 1
    only_extract = True

    basicConfig(level=INFO)
    path2data = Path(get_path_to_project()) / "bench_data"
    if not only_extract:
        if path2data.exists():
            rmtree(path2data, ignore_errors=True)

        for fs in sampling_rate_range:
            run_test_sec = 10
            print(f"Run trial with Sampling rate: {fs} Hz")
            run_experiment(
                sampling_rate=fs,
                do_batch=False,
                run_trial_sec=run_test_sec,
                repeat_trial=num_repetitions,
                folder_name=path2data.name
            )

    results = extract_results(path2data)
    plot_results(results)
