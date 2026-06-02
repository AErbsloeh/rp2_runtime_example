#include "hardware_io.h"
#include "callbacks/rpc_callbacks.h"
//#include "callbacks/fpga_callbacks.h"
#ifdef ADD_CYW43_SUPPORT
    #include "pico/cyw43_arch.h"
#endif


//#include "sens/w5500/w5500_udp.h"


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


     // Construct for W5500 Ethernet module configuration
    /*static spi_rp2_t w5500_spi_inst = {
    .spi_mod = spi0,
    .pin_mosi = 19,
    .pin_sclk = 18,
    .pin_miso = 16,
    .fspi_khz = 1000,
    };
    static w5500_udp_t w5500_config = {
        .spi = &w5500_spi_inst,
        .gpio_cs = 17,
        .gpio_rstn = 20,
        .gpio_intn = 21,
        .buffer_size = 2048,
        .udp_ip = {224, 0, 0, 5},
        .udp_port = 30000,
        .init_done = false,
    };
    w5500_udp_init(&w5500_config);
    w5500_udp_wait_until_connected(&w5500_config); 
    w5500_udp_print_info(&w5500_config);*/

    
    static bool valid_rpc[1] = {false};

    // Main Loop
    while (true) {  
        toggle_state_default_led();
        sleep_ms(1000);
        //w5500_udp_test(&w5500_config);
        // --- USB Protocol Handling --- 
        /*usb_handling_fifo_buffer(&usb_buffer);
        valid_rpc[0] = apply_rpc_callback(usb_buffer.data, usb_buffer.length, usb_buffer.ready);
        //valid_rpc[1] = apply_fpga_callback(usb_buffer.data, usb_buffer.length, usb_buffer.ready);
        if(!valid_rpc[0]){
            set_system_state(STATE_ERROR);
        }
        // --- Sending data in main ---
        daq_check_send_data(&daq_config_raw);*/
    };
}
