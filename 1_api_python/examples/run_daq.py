from api import DeviceAPI
from logging import basicConfig, DEBUG, INFO


if __name__ == "__main__":
    basicConfig(level=INFO)

    DeviceAPI().do_reset()
    dut = DeviceAPI()
    do_plot = True
    do_record = True
    do_process = False
    do_LoadPlotConfigFromJson = False
    do_SavePlotConfig = False

    if do_LoadPlotConfigFromJson:
        dut.load_plotconfigs_from_json("config.json")
    else:
        dut.set_liveplotter_config_manually(
            do_process=do_process
        )
    if do_SavePlotConfig:
        dut.save_plotconfigs_to_json("config.json")

    dut.start_daq(
        sampling_rate=1000.0,
        do_plot=do_plot,
        do_record=do_record,
        do_process=do_process,
        window_sec=10.0,
    )
    dut.stop_daq()