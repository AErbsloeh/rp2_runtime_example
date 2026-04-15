def _convert_pin_state(state: int) -> str:
    """Function for converting the pin state
    :param state:   Integer with pin state from MCU
    :return:        String with pin state
    """
    led_name = ['LED_USER']
    if state == 0:
        return 'NONE'
    else:
        ret_text = ''
        for idx, led in enumerate(led_name):
            if state & (1 << idx):
                ret_text += f'{led}' if len(ret_text) == 0 else f'+{led}'
        if ret_text == '':
            raise ValueError("Translated pin state is undefined")
        return ret_text


def _convert_system_state(state: int) -> str:
    """Function for converting the pin state
    :param state:   Integer with pin state from MCU
    :return:        String with pin state
    """
    state_name = ["ERROR", "RESET", "INIT", "IDLE", "TEST", "DAQ"]
    if not 0 <= state < len(state_name):
        raise ValueError(f'Invalid system state: {state}')
    return state_name[state]


def _convert_rp2_adc_value(raw: int) -> float:
    """Function for converting the RP2 ADC value from integer to float"""
    if raw >= 4095:
        val0 = 4095
    elif raw < 0:
        val0 = 0
    else:
        val0 = raw
    return val0 * 3.3 / 4095


def _convert_rp2_temp_value(raw: int) -> float:
    """Function for converting the RP2 temperatur value from integer to float"""
    volt = _convert_rp2_adc_value(raw)
    return 27 - (volt - 0.706) / 0.001721
