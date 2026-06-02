from api import DeviceAPI
from logging import basicConfig, INFO


if __name__ == "__main__":
    basicConfig(level=INFO)

    dut = DeviceAPI(
        transport="wifi",
        host="172.20.10.5",
        port=4242,
        timeout=0.5,
    )

    dut.define_channel_layout(channel_layout=[0, 1], channel_names=["CH0", "CH1"])

    dut.start_daq(
        sampling_rate=500.0,
        do_batch=True,
        do_plot=False,
        window_sec=4.0,
        folder_name="wifi_data",
    )
    dut.wait_daq(300)
    dut.stop_daq()
    dut.close()
