from api import DeviceAPI
from logging import basicConfig, DEBUG, INFO


if __name__ == "__main__":
    basicConfig(level=INFO)

    DeviceAPI().do_reset()
    dut = DeviceAPI()
    do_plot = True
    do_record = True
    do_process = True

    dut.start_daq(
        sampling_rate=1000.0,
        do_plot=do_plot,
        do_record=do_record,
        do_process=do_process,
        window_sec=30.0,
    )
    dut.stop_daq()