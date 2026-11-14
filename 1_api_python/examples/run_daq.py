from api import DeviceAPI
from logging import basicConfig, DEBUG, INFO


if __name__ == "__main__":
    basicConfig(level=INFO)

    DeviceAPI().do_reset()
    dut = DeviceAPI()
    do_plot = True
    do_record = True
    do_process = False

    dut.setup_live_plotter(
        plot_titles=[["plot1", "plot2"], [None,"plot3"]]
    )

    dut.add_plot_channel(
        lsl_stream_name="data",
        channel_name="CH0 Raw",
        channel_unit="LSB",
        curve_color='r',
        position=(0, 0),
        stream_channel_idx=0
    )
    dut.add_plot_channel(
        lsl_stream_name="data",
        channel_name="CH1 Raw",
        channel_unit="LSB",
        curve_color='b',
        position=(0, 0),
        stream_channel_idx=1
    )

    if do_process:
        dut.add_plot_channel(
            lsl_stream_name="filt",
            channel_name="CH0 Filtered",
            channel_unit="LSB",
            curve_color="#FF0000",  # Rot
            position=(0, 1),
            stream_channel_idx=0
        )
    dut.start_daq(
        sampling_rate=1000.0,
        do_plot=do_plot,
        do_record=do_record,
        do_process=do_process,
        window_sec=10.0,
    )
    dut.stop_daq()
