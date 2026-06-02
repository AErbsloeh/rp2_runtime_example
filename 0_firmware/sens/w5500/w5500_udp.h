#ifndef W5500_H_
#define W5500_H_


#include "hal/spi/spi.h"


/*! \brief Struct for configuring the module for W5500 Ethernet module on board
* \param spi            Pointer to SPI interface
* \param gpio_cs        Chip Select GPIO pin
* \param gpio_rstn      Reset GPIO pin
* \param gpio_intn      Interrupt GPIO pin
* \param buffer_size    Buffer size for Ethernet communication
* \param init_done      Initialization done flag
*/
typedef struct {
    spi_rp2_t *spi;
    uint8_t gpio_cs;
    uint8_t gpio_rstn;
    uint8_t gpio_intn;
    uint16_t buffer_size;
    uint8_t udp_ip[4];
    uint16_t udp_port;
    bool init_done;
} w5500_udp_t;


static spi_rp2_t w5500_spi_inst = {
    .spi_mod = spi0,
    .pin_mosi = 19,
    .pin_sclk = 18,
    .pin_miso = 16,
    .fspi_khz = 8000,
    .mode = 0,
    .msb_first = true,
    .init_done = false
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


/*! \brief Resetting the W5500 Ethernet module using SPI interface
* \param config    Pointer to W5500 Ethernet module configuration struct
* \return          None
*/
void w5500_udp_do_reset(w5500_udp_t *config);


/*! \brief Initialization of W5500 Ethernet module  using SPI interface
* \param config    Pointer to W5500 Ethernet module  configuration struct
* \return          Initialization success flag
*/
bool w5500_udp_init(w5500_udp_t *config);



bool w5500_udp_phy_connected(w5500_udp_t *config);


void w5500_udp_wait_until_connected(w5500_udp_t *config);


/*! \brief Printing network information of the W5500 Ethernet module using SPI interface
* \param config    Pointer to W5500 Ethernet module configuration struct
* \return          None
*/
void w5500_udp_print_info(w5500_udp_t *config);


/*! \brief Testing network connection of the W5500 Ethernet module using SPI interface
* \param config    Pointer to W5500 Ethernet module configuration struct
* \return          None
*/
void w5500_udp_test(w5500_udp_t *config);


#endif
