from api import DeviceAPI
from logging import basicConfig, DEBUG, INFO


if __name__ == '__main__':
    basicConfig(level=DEBUG)

    DeviceAPI().do_reset()
    dut = DeviceAPI()
    dut.start_daq(
        sampling_rate=500.,
        do_batch=False,
        do_plot=True,
        window_sec=4.,
        track_util=True
    )
    dut.wait_daq(0.5* 60)
    dut.stop_daq()
