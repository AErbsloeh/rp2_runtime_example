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

    #temporary debug prints
    config = dut._get_daq_characteristics()

    print("\n===== DAQ CONFIGURATION =====")
    print("send_batch:", config.send_batch)
    print("num_channels:", config.num_channels)
    print("num_samples:", config.num_samples)
    print("bytes_sample:", config.bytes_sample)
    print("num_bytes_total:", config.num_bytes_total)
    print("expected_without_crc:", config.expected_bytes_without_crc)
    print("crc_bytes:", config.crc_bytes)
    print("expected_with_crc:", config.expected_bytes_without_crc + 2)
    print("=============================\n")

    print("layout:", dut._DeviceAPI__layout_channels)
    print("labels:", dut._DeviceAPI__layout_labels)

    dut.start_daq(
        sampling_rate=500.0,
        do_plot=True,
        do_process=False,
        do_record=False,
        window_sec=4.0,
    )
    dut.wait_daq(30)
    dut.stop_daq()

