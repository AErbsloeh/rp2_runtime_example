#include "peri/w5500/w5500_udp.h"
#include "wizchip_conf.h"
#include "wizchip_spi.h"
#include "socket.h"
#include "multicast.h"
#include <stdio.h>


static uint8_t wizchip_read(void) {
    uint8_t rx_data = 0x00;
    uint8_t tx_data = 0xFF;
    spi_read_blocking(spi0, tx_data, &rx_data, 1);
    return rx_data;
}


static void wizchip_write(uint8_t tx_data) {
    spi_write_blocking(spi0, &tx_data, 1);
}


static inline void wizchip_select(void) {
    gpio_put(17, false);
}


static inline void wizchip_deselect(void) {
    gpio_put(17, true);
}


static uint8_t ethernet_buf[2048] = {
    0,
};


static wiz_NetInfo g_net_info = {
    .mac = {0x00, 0x08, 0xDC, 0x12, 0x34, 0x56},    // MAC address
    .ip = {192, 168, 11, 2},                        // IP address
    .sn = {255, 255, 255, 0},                       // Subnet Mask
    .gw = {192, 168, 11, 1},                        // Gateway
    .dns = {8, 8, 8, 8},                            // DNS Server
    .dhcp = NETINFO_STATIC                          // DHCP enable/disable
};


bool w5500_udp_init(w5500_udp_t *config){
    // Set as input for testing if card is present
    gpio_init(config->gpio_cs);
    gpio_pull_up(config->gpio_cs);
    gpio_set_dir(config->gpio_cs, GPIO_OUT);
    gpio_put(config->gpio_cs, true);

    gpio_init(config->gpio_rstn);
    gpio_pull_up(config->gpio_rstn);
    gpio_set_dir(config->gpio_rstn, GPIO_OUT);
    gpio_put(config->gpio_rstn, false);

    gpio_init(config->gpio_intn);
    gpio_pull_up(config->gpio_intn);
    gpio_set_dir(config->gpio_intn, GPIO_IN);
    sleep_ms(1);

    if(!config->spi->init_done){
        configure_spi_module(config->spi, false);
    }

    wizchip_cris_initialize();
    w5500_udp_do_reset(config);

    // --- INIT PHASE
    /* CS function register */
    reg_wizchip_cs_cbfunc(wizchip_select, wizchip_deselect);
    /* SPI function register */
    reg_wizchip_spi_cbfunc(wizchip_read, wizchip_write);

    /* W5x00, W6x00 initialize */
    uint8_t memsize[2][8] = {{2, 2, 2, 2, 2, 2, 2, 2}, {2, 2, 2, 2, 2, 2, 2, 2}};
    if (ctlwizchip(CW_INIT_WIZCHIP, (void *)memsize) == -1) {
        config->init_done = false;
        return false;
    } else {
        network_initialize(g_net_info);
        config->init_done = true;
    }
    return config->init_done;
};


void w5500_udp_do_reset(w5500_udp_t *config){
    gpio_put(config->gpio_rstn, false);
    sleep_ms(100);
    gpio_put(config->gpio_rstn, true);
    sleep_ms(100);
    config->init_done = false;
}


bool w5500_udp_phy_connected(w5500_udp_t *config){
    uint8_t temp;
    if (config->init_done) {
        ctlwizchip(CW_GET_PHYLINK, &temp);
        return temp == PHY_LINK_ON;
    }
}


void w5500_udp_wait_until_connected(w5500_udp_t *config){
    uint8_t temp;
    if (config->init_done)    
        do {
            sleep_ms(100);
            if (ctlwizchip(CW_GET_PHYLINK, (void *)&temp) == -1) {
                return;
            }
        } while (temp == PHY_LINK_OFF);
}


void w5500_udp_print_info(w5500_udp_t *config){
    print_network_information(g_net_info);
}


void w5500_udp_test(w5500_udp_t *config){
    if(config->init_done){
        multicast_recv(0, ethernet_buf, config->udp_ip, config->udp_port);
    }
}