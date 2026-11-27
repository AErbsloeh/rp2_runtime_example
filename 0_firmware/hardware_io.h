#ifndef HARDWARE_IO_H_
#define HARDWARE_IO_H_


#include "hardware/gpio.h"
#include "hardware/clocks.h"
#include "hal/helper/helper.h"
#include "hal/led/led.h"
#include "hal/tmr/tmr.h"
#include "hal/adc/adc.h"
#include "hal/daq/daq.h"
#include "hal/daq/processing.h"
#include "hal/transport/transport.h"

#include "src/init_system.h"
#include "src/user_logic.h"

extern system_state_t system_state;
// ==================== PIN DEFINITION =====================
#define BUTTON_BOARD        11
#define LED_DEFAULT         25
#define FPGA_EN_POWER_GPIO  23

// ==================== I2C DEFINITION =====================
//extern i2c_rp2_t i2c_mod;


// ==================== SPI DEFINITION =====================
//extern spi_rp2_t spi_mod;

// --- FPGA flash and FPGA logic on one shared SPI bus
// Needs peri_fpga_flash / peri_fpga_logic in CMakeLists.txt, the definitions are in hardware_io.c
//#include "peri/fpga/flash.h"
//#include "peri/fpga/logic.h"
//#include "peri/fpga/fpga_spi_shared.h"
//extern spi_rp2_t spi_fpga;            //!< SPI bus on the configuration pins of the FPGA (flash programming and logic access)
//extern fpga_flash_t flash_env5;       //!< FPGA flash: programming of the bitstream (FPGA in reset)
//extern fpga_logic_t fpga_env5;        //!< FPGA logic: SPI middleware for the data exchange (FPGA configured, flash deselected)
//extern fpga_spi_shared_t fpga_bus;    //!< Shared bus: attaches the pins to the logic and releases them for the flash


// ================ PICO/SYSTEM DEFINITION =================
// --- Internal Temp Sensor
extern rp2_adc_t adc_temp;

// --- USB Communication
extern transport_rx_buffer_t rx_buffer;

// --- DAQ Sampling
extern daq_data_t daq_config0;
extern daq_data_t daq_config1;
extern daq_data_t* const daq_streams[];
extern const uint8_t daq_num_streams;
extern tmr_repeat_irq_t tmr_daq0_hndl;

// --- Data processing on core 1
extern const processing_config_t processing_config;

#endif
