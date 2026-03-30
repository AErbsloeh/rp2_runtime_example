#ifndef HARDWARE_IO_H_
#define HARDWARE_IO_H_


#include "hardware/gpio.h"
#include "hardware/clocks.h"
#include "hal/led/led.h"
#include "hal/tmr/tmr.h"
#include "hal/usb/usb.h"
#include "hal/adc/adc.h"
#include "hal/daq/daq.h"

#include "src/init_system.h"
#include "src/testbench.h"


extern system_state_t system_state;
// ==================== PIN DEFINITION =====================
#define BUTTON_BOARD        11
#define LED_DEFAULT    25

// ==================== I2C DEFINITION =====================
//extern i2c_rp2_t i2c_mod;


// ==================== SPI DEFINITION =====================
//extern spi_rp2_t spi_mod;


// ================ PICO/SYSTEM DEFINITION =================
// --- Internal Temp Sensor
extern rp2_adc_t adc_temp;

// --- USB Communication
extern usb_rp2_t usb_buffer;

// --- DAQ Sampling
extern daq_data_t daq_sample_data;
extern tmr_repeat_irq_t tmr_daq0_hndl;

#endif
