#include "hardware_io.h"
#include "callbacks/rpc_callbacks.h"
#include "src/transport.h"
// #include "callbacks/fpga_callbacks.h"
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
    
    static bool valid_rpc[1] = {false};

    // Main Loop
    while (true)
    {
        // --- USB Protocol Handling ---
        transport_poll_rx(&rx_buffer);
        valid_rpc[0] = apply_rpc_callback(rx_buffer.data, rx_buffer.length, rx_buffer.ready);
        // valid_rpc[1] = apply_fpga_callback(usb_buffer.data, usb_buffer.length, usb_buffer.ready);
        if (!valid_rpc[0])
        {
            set_system_state(STATE_ERROR);
        }
        // --- Sending data in main ---
        daq_check_send_data(&daq_config_raw);
    };
}
