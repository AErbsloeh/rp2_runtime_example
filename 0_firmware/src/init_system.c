#include "src/init_system.h"
#include "hardware/watchdog.h"
#include "hardware_io.h"
#include "callbacks/gpio_callbacks.h"


void reset_pico_mcu(bool wait_until_done){
    set_system_state(STATE_RESET);
    watchdog_enable(100, 1);
    if(wait_until_done){
        while(true){
            tight_loop_contents();
        }
    };
}


bool init_gpio_pico(bool block_usb){
    set_system_state(STATE_INIT);
    
    // --- Init of GPIOs
    //set_gpio_default_led(LED_PIN_DEFAULT); // Activate only for custom boards (KB2040 -> WS2812, PICO1/2_W -> CYW43)
    init_default_led();

    // --- Init GPIO + IRQ (Low Level)
    /*gpio_init(BUTTON_BOARD);
    gpio_set_dir(BUTTON_BOARD, GPIO_IN);
    gpio_pull_up(BUTTON_BOARD);
    gpio_set_slew_rate(BUTTON_BOARD, GPIO_SLEW_RATE_SLOW);
    gpio_set_irq_enabled_with_callback(BUTTON_BOARD, GPIO_IRQ_EDGE_FALL, true, &irq_gpio_callbacks);*/

    // --- Init of Serial COM-Port
    usb_init(&usb_buffer);
    if(block_usb){
        usb_wait_until_connected();
    }
    sleep_ms(1000);
    return true;
}


bool init_system(void){
    uint8_t num_init_done = 0;
	
	// --- Internal ADC
	if(rp2_adc_init(&adc_temp))
		num_init_done++;

    if(daq_init_sampling(&tmr_daq0_hndl, &daq_config_raw))
        num_init_done++;

    // --- Blocking Routine if init is not completed
    sleep_ms(10);
    if(num_init_done == 2){
        set_system_state(STATE_IDLE);
        return true;
    } else {
        set_system_state(STATE_ERROR);
        while(true){
            sleep_ms(100);
            toggle_state_default_led();
        }
        return false;
    }
}


system_state_t get_system_state(void){
    return system_state;
}


bool set_system_state(system_state_t new_state){
    bool valid_state = false;
    
    if((system_state != new_state)){
        system_state = new_state;
        switch(new_state){
            case STATE_INIT:
                set_state_default_led(false);
                valid_state = true;
                break;
            case STATE_IDLE:
                set_state_default_led(true);
                valid_state = true;
                break;
            case STATE_ERROR:
                set_state_default_led(true);
                valid_state = false;
                break;
            default:
                set_state_default_led(false);
                valid_state = false;
                break;
        }
        return valid_state;
    } else {
        return false;
    };    
}
