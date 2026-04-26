from api import DeviceAPI
from logging import basicConfig, DEBUG, INFO


if __name__ == '__main__':
    basicConfig(level=INFO)

    DeviceAPI().do_reset()
    dut = DeviceAPI()
    dut.define_channel_layout(
        channel_layout=[0, 1],
        channel_names=['CH0', 'CH1']
    )
    dut.start_daq(
        sampling_rate=500.,
        do_batch=True,
        do_plot=True,
        window_sec=4.,
        track_util=True
    )
    dut.wait_daq(0.5* 60)
    dut.stop_daq()
