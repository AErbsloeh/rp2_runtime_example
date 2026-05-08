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
        // --- USB Protocol Handling --- 
        usb_handling_fifo_buffer(&usb_buffer);
        apply_rpc_callback(usb_buffer.data, usb_buffer.length, usb_buffer.ready);

        // --- Sending data in main ---
        if(daq_check_send_data(&daq_config_raw))
            toggle_state_default_led();
    };
}

// ----------------- CODE FOR W5500 TESTING -----------------
/*#include "peri/w5500/w5500_udp.h"
#include <stdio.h>


int main(){   
    // Init Phase 
    stdio_init_all();
    sleep_ms(3000);
    printf("a\n");

    // Construct for W5500 Ethernet module configuration
    static spi_rp2_t w5500_spi_inst = {
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
    w5500_udp_wait_until_connected(&w5500_config); // Wait until the Ethernet cable is connected (with a timeout of 10 seconds)
    w5500_udp_print_info(&w5500_config); // Read back the configuration information and print it
    
    // Main Loop
    while (true) {  
        w5500_udp_test(&w5500_config);
    }
}*/

// ----------------- TEMPLATE CODE TO BUILD AN EXAMPLE -----------------
/*#include <stdio.h>
#include "hal/led/led.h"

int main(){   
    // Init Phase
    init_default_led(); 
    stdio_init_all();
    sleep_ms(3000);

    // Pre-Phase
    set_state_default_led(true);
    printf("a\n");

    // Main Loop
    while (true) {  
        sleep_ms(1000);
        printf("Hello, World!\n");
        toggle_state_default_led();
    }
}*/
