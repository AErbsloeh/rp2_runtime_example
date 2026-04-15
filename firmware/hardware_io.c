#include "hardware_io.h"


system_state_t system_state = STATE_ERROR;
// ==================== I2C DEFINITION =====================
/*spi_rp2_t spi_mod = {
	.pin_mosi = PICO_DEFAULT_SPI_TX_PIN,
    .pin_sclk = PICO_DEFAULT_SPI_SCK_PIN,
    .pin_miso = PICO_DEFAULT_SPI_RX_PIN,
    .spi_mod = PICO_DEFAULT_SPI,
    .fspi_khz = 1000,
    .mode = 0,
    .msb_first = true,
    .init_done = false
};*/


// ==================== SPI DEFINITION =====================
/*i2c_rp2_t i2c_mod = {
	.pin_sda = PICO_DEFAULT_I2C_SDA_PIN,
	.pin_scl = PICO_DEFAULT_I2C_SCL_PIN,
	.i2c_mod = i2c0,
	.fi2c_khz = 100,
	.avai_devices = 0,
	.init_done = false	
};*/


// =============== PICO/SYSTEM DEFINITION ==================
// --- Internal Temp Sensor
rp2_adc_t adc_temp = {
    .adc_channel = RP2_ADC_TEMP,
    .sampling_rate = 1,
    .buffersize = 1,
    .use_dma = false,
    .init_done = false
};

// --- USB PROTOCOL
usb_rp2_t usb_buffer = {
	.ready = false,
	.length = 3,
	.position = 2
};  

// --- DAQ Sampling
fifo_t daq_fifo = {
    .element_size = sizeof(uint16_t)
};
daq_data_t daq_config_raw = {
    .packet_id = 0xA0,
    .iteration = 0,
    .runtime_first = 0,
    .runtime_last = 0,
    .num_channels = 2,
    .num_samples = 16,
    .data = &daq_fifo,
    .send_batch = true,
    .new_data = false
};

uint16_t data[2] = {0, 0};
bool irq_tmr_daq0(repeating_timer_t *rt){
    data[0] += 16;
    data[1] = rp2_adc_read_raw(&adc_temp);
    return daq_irq_process(&daq_config_raw, data);
};
repeating_timer_t tmr_daq0;
tmr_repeat_irq_t tmr_daq0_hndl = {
    .timer = &tmr_daq0,
    .irq_number = 0,
    .period_us = -250000,
    .alarm_done = false,
    .enable_state = false,
    .init_done = false,
    .func_irq = irq_tmr_daq0
};
