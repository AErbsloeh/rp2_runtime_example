from api import DeviceAPI
from logging import basicConfig, DEBUG, INFO


if __name__ == "__main__":
    basicConfig(level=INFO)

    DeviceAPI().do_reset()
    dut = DeviceAPI()
    dut.define_channel_layout(
        channel_layout=[1, 2],
        channel_names=["CH0", "CH1"],
    )
    dut.start_daq(
        sampling_rate=100.0,
        do_plot=False,
        do_record=True,
        do_process=False,
        window_sec=4.0,
    )
    dut.wait_daq(60)
    dut.stop_daq()
