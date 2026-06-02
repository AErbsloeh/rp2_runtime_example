#include "peri/w5500/w5500_udp.h"
#include <stdio.h>


int main(){   
    // Init Phase 
    stdio_init_all();

    sleep_ms(3000);
    printf("a");

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
    print_network_information(g_net_info); // Read back the configuration information and print it

    // Main Loop
    while (true) {  
        w5500_udp_test(&w5500_config);    
    }
}
