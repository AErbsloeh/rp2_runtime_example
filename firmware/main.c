#include "hardware_io.h"
#include "callbacks/rpc_callbacks.h"
#ifdef ADD_CYW43_SUPPORT
    #include "pico/cyw43_arch.h"
#endif

int main(){   
    #ifdef ADD_CYW43_SUPPORT
        if (cyw43_arch_init()) {
            return -1;
        }
    #endif

    // Init Phase 
    init_gpio_pico(false);
    init_system();
    run_testbench(TB_NONE);    

    // Main Loop
    while (true) {  
        /* --- USB Protocol Handling --- */
        usb_handling_fifo_buffer(&usb_buffer);
        apply_rpc_callback(usb_buffer.data, usb_buffer.length, usb_buffer.ready);

        /*--- Sending data in main */
        if(daq_sample_data.send_batch){
            if(daq_is_fifo_full(&daq_sample_data)){
                daq_sample_data.iteration ++;
                daq_send_data_usb(&daq_sample_data, daq_sample_data.data->length);
                toggle_state_default_led();
            };    
        } else {
            if(daq_sample_data.new_data){
                daq_sample_data.new_data = false;  
                daq_sample_data.iteration ++;
                daq_send_data_usb(&daq_sample_data, 2);
                toggle_state_default_led();
            };
        }
    };
}
