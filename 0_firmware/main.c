#include "hardware_io.h"
#include "callbacks/rpc_callbacks.h"
#ifdef ADD_FPGA_SUPPORT
    #include "callbacks/fpga_callbacks.h"
#endif
#ifdef ADD_CYW43_SUPPORT
    #include "pico/cyw43_arch.h"
#endif


int main(){
    #ifdef ADD_CYW43_SUPPORT
        if (cyw43_arch_init()){
            return -1;
        }
    #endif
    // --- Init Phase
    init_gpio_pico(false);
    bool valid_rpc = init_system();
    // --- Main Loop
    while (true){
        // Processing USB RX data
        transport_poll_rx(&rx_buffer);
        valid_rpc &= apply_rpc_callback(&rx_buffer);
        #ifdef ADD_FPGA_SUPPORT
            valid_rpc &= apply_fpga_callback(&rx_buffer);
        #endif
        if (!valid_rpc){
            set_system_state(STATE_ERROR);
        }
        // Sending DAQ data
        daq_check_send_data(&daq_config_raw);
    };
}
