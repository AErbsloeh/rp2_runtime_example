#include "callbacks/gpio_callbacks.h"
#include "hardware_io.h"


// ============================== ISR ROUTINES ==============================
void irq_gpio_button_user(uint gpio, uint32_t events){
    toggle_state_default_led();
}


void irq_gpio_callbacks(uint gpio, uint32_t events){
    switch(gpio){
        case BUTTON_BOARD:  irq_gpio_button_user(gpio, events);                 break;
		default:			set_system_state(STATE_ERROR);				        break;
    };
}
