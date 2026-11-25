#include "hardware_io.h"
#include "stdlib.h"

system_state_t system_state = STATE_ERROR;
// ==================== I2C DEFINITION =====================
/*i2c_rp2_t i2c_mod = {
	.pin_sda = PICO_DEFAULT_I2C_SDA_PIN,
	.pin_scl = PICO_DEFAULT_I2C_SCL_PIN,
	.i2c_mod = i2c0,
	.fi2c_khz = 100,
	.avai_devices = 0,
	.init_done = false	
};*/


// ==================== SPI DEFINITION =====================
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
transport_rx_buffer_t rx_buffer = {
	.ready = false,
	.length = 3,
	.position = 2,
    .data = NULL
};  

// =============== Registry of all DAQ streams (order = index used by GET_CHARAC_DAQ) ===============
// --- DAQ Sampling Processes
fifo_t a = {};
fifo_t b = {};
daq_data_t daq_config0 = {
    .packet_id = 0xA0,
    .packet_tail = 0xFF,
    .iteration = 0,
    .runtime_first = 0,
    .runtime_last = 0,
    .is_signed = false,
    .element_size = sizeof(uint16_t),
    .num_channels = 2,
    .num_samples = 32,
    .send_mode = DAQ_MODE_SAMPLE,
    .new_data = false,
    .data0 = &a,
    .data1 = &b
};

fifo_t c = {};
fifo_t d = {};
daq_data_t daq_config1 = {
    .packet_id = 0xFA,
    .packet_tail = 0xBC,
    .iteration = 0,
    .runtime_first = 0,
    .runtime_last = 0,
    .is_signed = false,
    .element_size = sizeof(uint16_t),
    .num_channels = 1,
    .num_samples = 1024,
    .send_mode = DAQ_MODE_BUFFER_SINGLE,
    .new_data = false,
    .data0 = &c,
    .data1 = &d
};

daq_data_t* const daq_streams[] = {&daq_config0, &daq_config1};
const uint8_t daq_num_streams = sizeof(daq_streams) / sizeof(daq_streams[0]);

// --- IRQ Routine for DAQ sampling
uint16_t data_raw0[2] = {0, 0};
uint16_t data_raw1[1] = {0};
bool irq_tmr_daq0(repeating_timer_t *rt){
    data_raw0[0] += 16;
    data_raw0[1] = rp2_adc_read_raw(&adc_temp);

    data_raw1[0] -= 64;

    daq_irq_process(&daq_config0, data_raw0);
    daq_irq_process(&daq_config1, data_raw1);
    return true;
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
