from api import DeviceAPI
from logging import basicConfig, INFO
from time import sleep


if __name__ == "__main__":
    basicConfig(level=INFO)

    dut = DeviceAPI(
        transport="wifi",
        host="172.20.10.5",
        port=4242,
        timeout=0.5,
    )

    is_running = False
    try:
        dut.start_daq(
            sampling_rate=500.0,
            do_plot=False,
            do_record=False,
            do_process=True,
        )
        is_running = True
        while True:
            sleep(1.)
    except KeyboardInterrupt:
        pass
    finally:
        if is_running:
            dut.stop_daq()
        dut.close()
