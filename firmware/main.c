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
        if(daq_check_send_data(&daq_sample_data))
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


// ----------------- TEMPLATE CODE TO BUILD AN EXAMPLE -----------------
/*#include <stdio.h>
#include "hal/led/led.h"
#include "hal/spi/spi.h"
#include "peri/fpga/flash.h"
#include "peri/fpga/fpga_spi.h"
#include "peri/fpga/fpga_config.h"


#define FPGA_EN_POWER_GPIO  23


int main(){   
    // Init Phase
    init_default_led(); 

     // Pin Init for FPGA Power Enable
    gpio_init(FPGA_EN_POWER_GPIO);
    gpio_set_dir(FPGA_EN_POWER_GPIO, GPIO_OUT);
    gpio_put(FPGA_EN_POWER_GPIO, false);

    // SPI


    stdio_init_all();
    sleep_ms(3000);

    // Pre-Phase
    if(fpga_program_init(&flash_env5))
        printf("FPGA Flash initialization done!\n");

    if(fpga_spi_init(&fpga_env5))
        printf("FPGA SPI initialization done!\n");

    set_state_default_led(true);

    uint16_t ret = fpga_flash_get_id(&flash_env5);
    printf("%x\n", ret);
    sleep_ms(1);
    
    uint8_t data_rx[256];
    fpga_program_flash_read_data(&flash_env5, 0, data_rx, sizeof(data_rx));
    printf("\n");
    printf("Read flash content (256 bytes) before change:\n");
    for(uint16_t idx=0; idx < 256; idx++){
        printf("%x ", data_rx[idx]); 
        if(idx % 8 == 0 && idx > 0)
            printf("\n");
    }    
    printf("\n");

    data_rx[0] = 0xff;
    data_rx[1] = 0x04;
    data_rx[2] = 0xff;
    fpga_program_flash_write_data(&flash_env5, 0, data_rx, sizeof(data_rx));
    sleep_ms(1);
    printf("\n");
    printf("Read flash content (256 bytes) after change:\n");
    for(uint16_t idx=0; idx < 256; idx++){
        printf("%x ", data_rx[idx]); 
        if(idx % 8 == 0 && idx > 0)
            printf("\n");
    }    
    printf("\n");    
    
    // Enable power to FPGA
    printf("Enablng FPGA power!\n");
    gpio_put(FPGA_EN_POWER_GPIO, true);
    sleep_ms(1000);    

    // Main Loop
    printf("Hello, World!\n");

    uint8_t data_rx0[3] = {0};
    uint8_t data_toggle_led[3] = {0x08, 0x00, 0x00};
    uint16_t cnt = 0;
    bool state = false;
    fpga_spi_reset_do(&fpga_env5, state);

    while (true) {  
        sleep_ms(250);
        fpga_spi_send_data(&fpga_env5, data_toggle_led, data_rx0);
        printf("%x %x %x\n", data_rx0[0], data_rx0[1], data_rx0[2]);
        //toggle_state_default_led();

        fpga_spi_reset_do(&fpga_env5, state);
        state = !state;
        cnt++;
        if(cnt >= 200){
            cnt = 0;
            fpga_program_reset_do(&flash_env5);
        }
    }
}*/
