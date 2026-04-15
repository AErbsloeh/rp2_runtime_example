from api import DeviceAPI


if __name__ == '__main__':
    DeviceAPI().do_reset()
    dut = DeviceAPI()
    dut.update_daq_sampling_rate(10000.)

    dut.start_daq(
        do_batch=True,
        do_plot=True,
        window_sec=10.,
        track_util=True
    )
    dut.wait_daq(10.)
    dut.stop_daq()
