from api import DeviceAPI
from logging import basicConfig, DEBUG, INFO


if __name__ == "__main__":
    basicConfig(level=INFO)

    DeviceAPI().do_reset()
    dut = DeviceAPI()
    dut.define_channel_layout(
        channel_layout=[0, 1],
        channel_names=["CH0", "CH1"],
    )
    print("layout:", dut._DeviceAPI__layout_channels)
    print("labels:", dut._DeviceAPI__layout_labels)

    dut.start_daq(
        sampling_rate=500.0,
        do_plot=True,
        do_process=False,
        do_record=False,
        window_sec=4.0,
    )
   # dut.wait_daq(30)
   # dut.stop_daq()


