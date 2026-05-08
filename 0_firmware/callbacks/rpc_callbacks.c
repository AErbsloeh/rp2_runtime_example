#include "callbacks/rpc_callbacks.h"
#include "hardware_io.h"
#include "version.h"


// ============================= COMMANDS =============================
typedef enum {
    ECHO = 0,
    RESET,
    GET_SYSTEM_STATE,
    GET_CHARAC_DAQ,
    ENABLE_LED,
    DISABLE_LED,
    TOGGLE_LED,
    START_DAQ,
    STOP_DAQ,
    SET_PERIOD_DAQ,
    SET_BATCH_DAQ,
    USB_CMD_COUNT   // Just for getting the number of commands, not an actual command
} usb_cmd_t;


// ========================== PROCOTOL FUNCS ==========================
void echo(char* buffer, size_t length){
    usb_send_bytes(buffer, length);
}


void system_reset(void){
    reset_pico_mcu(true);
}


void get_charac_system(void){
    uint64_t runtime = get_runtime_ms();
    uint16_t temp_raw = rp2_adc_read_raw(&adc_temp);
    uint16_t clk_val = (uint16_t)(clock_get_hz(clk_sys) / 10000);

    char buffer_send[19] = {GET_SYSTEM_STATE};
    buffer_send[1] = (uint8_t)(get_system_state() >> 0);
    buffer_send[2] = (uint8_t)(get_system_state() >> 8);
    buffer_send[3] = (uint8_t)(clk_val >> 0);
    buffer_send[4] = (uint8_t)(clk_val >> 8);
    /* PIN STATE for byte 5 & 6 */
    buffer_send[5] = (uint8_t)((get_state_default_led() << 0x00) | (gpio_get(23) << 0x01));
    buffer_send[6] = 0x00;
    buffer_send[7] = (uint8_t)(temp_raw >> 0);
    buffer_send[8] = (uint8_t)(temp_raw >> 8);
    buffer_send[9] = PROGRAM_VERSION[0]-48;
    buffer_send[10] = PROGRAM_VERSION[2]-48;
    for(uint8_t idx = 0; idx < 8; idx++){
        buffer_send[idx+11] = (uint8_t)runtime;
        runtime >>= 8;
    }
    usb_send_bytes(buffer_send, sizeof(buffer_send));
}


void get_charac_daq(void){
    daq_data_t* data = &daq_config_raw;
    tmr_repeat_irq_t* timer = &tmr_daq0_hndl;

    uint16_t num_bytes_sample = daq_get_number_bytes_per_sample(data);
    uint16_t num_bytes_batch = daq_get_number_bytes_per_batch(data);
    uint16_t num_bytes = (data->send_batch) ? num_bytes_batch : num_bytes_sample;
    uint16_t num_samples = (data->send_batch) ? data->num_samples : 1;
    int64_t runtime = timer->period_us;

    char buffer_send[20] = {GET_CHARAC_DAQ};
    buffer_send[1] = (uint8_t)(data->packet_id);
    buffer_send[2] = 0xFF;  // tail command
    buffer_send[3] = 0x00;  // is sample signed datatype? 
    buffer_send[4] = (uint8_t)(data->send_batch);
    buffer_send[5] = (uint8_t)(data->num_channels >> 0);
    buffer_send[6] = (uint8_t)(data->num_channels >> 8);
    buffer_send[7] = (uint8_t)(num_samples >> 0);
    buffer_send[8] = (uint8_t)(num_samples >> 8);
    buffer_send[9] = (uint8_t)(num_bytes >> 0);
    buffer_send[10] = (uint8_t)(num_bytes >> 8);
    buffer_send[11] = (uint8_t)(data->data->element_size);
    for(uint8_t idx = 0; idx < 8; idx++){
        buffer_send[12+idx] = (uint8_t)runtime;
        runtime >>= 8;
    }
    usb_send_bytes(buffer_send, sizeof(buffer_send));   
}


void enable_led(void){
    set_state_default_led(true);
}


void disable_led(void){
    set_state_default_led(false);
}


void toggle_led(void){
    toggle_state_default_led();
}


void start_daq(void){
    daq_start_sampling(&tmr_daq0_hndl);
    set_system_state(STATE_DAQ);
}


void stop_daq(void){
    set_system_state(STATE_IDLE);
    daq_stop_sampling(&tmr_daq0_hndl);
}


void update_period_daq(char* buffer){
    float new_sampling_rate_hz = (buffer[1] << 8) | (buffer[2] << 0);
    int64_t new_rate_us = (float)-1000000 / new_sampling_rate_hz;
    daq_update_sampling_rate(&tmr_daq0_hndl, new_rate_us);
}


void set_batch_daq(char* buffer){
    daq_config_raw.send_batch = (buffer[2] == 0x01);
}



// ======================== FLASH / FPGA CMDS ==========================
#include "peri/fpga/flash.h"
#include "peri/fpga/fpga_spi.h"
#include "peri/fpga/fpga_config.h"


#define FPGA_EN_POWER_GPIO 23

static flash_fpga_t *flash_config = &flash_env5;
static uint8_t flash_buffer_data[256] = {0};
static uint32_t flash_address = 0;
static uint16_t flash_page_position = 0; 


typedef enum {
    FPGA_INIT_PIN = USB_CMD_COUNT,
    FPGA_POWER_STATE,
    FLASH_INFOS,
    FLASH_START_ERASE_ALL,
    FLASH_START_ERASE_SEC,
    FLASH_CHECK_ERASE,
    FLASH_SET_ADDR_UPPER,
    FLASH_SET_ADDR_LOWER,
    FLASH_GET_ADDR,
    FLASH_READ_DATA,
    FLASH_WRITE_BUFFER,
    FLASH_WRITE_DATA,
    FPGA_DO_RESET,
    FPGA_SEND_DATA
} fpga_cmd_t;


void init_pin_fpga(void){
    gpio_init(FPGA_EN_POWER_GPIO);
    gpio_set_dir(FPGA_EN_POWER_GPIO, GPIO_OUT);
    gpio_put(FPGA_EN_POWER_GPIO, false);

    uint8_t num_init = 0;
    if(fpga_program_init(flash_config))
        num_init++;

    if(fpga_spi_init(&fpga_env5))
        num_init++;

    char buffer_send[3] = {FPGA_INIT_PIN};
    buffer_send[1] = 0x00;
    buffer_send[2] = num_init == 0x02;
    usb_send_bytes(buffer_send, sizeof(buffer_send));
}


void set_state_fpga_power(char* buffer){
    bool state = (buffer[2] == 0x01);
    gpio_put(FPGA_EN_POWER_GPIO, state);
}


void get_flash_infos(void){
    uint8_t manu_id = fpga_flash_get_manufacturer_id(flash_config);
    uint16_t device_id = fpga_flash_get_device_id(flash_config);
    uint16_t jedec_id = fpga_flash_get_jedec_id(flash_config);
    uint16_t status_reg = fpga_flash_get_status_register(flash_config);

    char buffer_send[12] = {FLASH_INFOS};
    buffer_send[1] = (uint8_t)(manu_id >> 0);
    buffer_send[2] = (uint8_t)(device_id >> 8);
    buffer_send[3] = (uint8_t)(device_id >> 0);
    buffer_send[4] = (uint8_t)(jedec_id >> 8);
    buffer_send[5] = (uint8_t)(jedec_id >> 0);
    buffer_send[6] = (uint8_t)(status_reg >> 8);
    buffer_send[7] = (uint8_t)(status_reg >> 0);
    buffer_send[8] = (uint8_t)(flash_config->page_size >> 0);
    buffer_send[9] = (uint8_t)(flash_config->page_size >> 8);
    buffer_send[10] = (uint8_t)(flash_config->block_size >> 0);
    buffer_send[11] = (uint8_t)(flash_config->block_size >> 8);
    usb_send_bytes(buffer_send, sizeof(buffer_send));
}


void start_erasing_flash_all(void){
    set_system_state(STATE_ERASE_FLASH);
    fpga_flash_erasing_all_start(flash_config);
}


void start_erasing_flash_sector(void){
    set_system_state(STATE_ERASE_FLASH);
    fpga_flash_erasing_sector_start(flash_config, flash_address);
}


void check_erasing_flash(void){
    bool state = false;
    if(get_system_state() == STATE_ERASE_FLASH){
        if(fpga_flash_erasing_is_done(flash_config)){
            state = true;
            fpga_flash_erasing_stop(flash_config);
            set_system_state(STATE_IDLE);
        }
    }

    char buffer_send[3] = {FLASH_CHECK_ERASE};
    buffer_send[1] = 0x00;
    buffer_send[2] = (uint8_t)(state);
    usb_send_bytes(buffer_send, sizeof(buffer_send));
}


void set_flash_starting_address_upper(char* buffer){
    uint16_t upper = (buffer[1] << 8) | buffer[2];
    flash_address = (uint32_t)(upper << 16);
}


void set_flash_starting_address_lower(char* buffer){
    uint16_t lower = (buffer[1] << 8) | buffer[2];
    flash_address |= lower;
}


void get_flash_starting_address(void){
    char buffer_send[1 + sizeof(flash_address)] = {FLASH_GET_ADDR};
    for(size_t idx = 0; idx < sizeof(flash_address); idx++){
        buffer_send[1 + idx] = (uint8_t)(flash_address >> (idx * 8));
    }
    usb_send_bytes(buffer_send, sizeof(buffer_send));
}


void read_flash_data(void){
    fpga_flash_read_data(flash_config, flash_address, flash_buffer_data, sizeof(flash_buffer_data));

    char buffer_send[1 + sizeof(flash_buffer_data)] = {FLASH_READ_DATA};
    for(size_t idx = 0; idx < sizeof(flash_buffer_data); idx++){
        buffer_send[1 + idx] = (uint8_t)(flash_buffer_data[idx]);
    }
    usb_send_bytes(buffer_send, sizeof(buffer_send));
}


void write_data_into_flash_buffer(char* buffer){    
    flash_buffer_data[flash_page_position++] = buffer[1];   
    flash_buffer_data[flash_page_position++] = buffer[2];
}


void write_buffer_into_flash(void){
    bool state = fpga_flash_write_data(flash_config, flash_address, flash_buffer_data, sizeof(flash_buffer_data));
    flash_page_position = 0;

    char buffer_send[3] = {FLASH_WRITE_DATA};
    buffer_send[1] = 0x00;
    buffer_send[2] = (uint8_t)(state);
    usb_send_bytes(buffer_send, sizeof(buffer_send));
}


void fpga_do_reset(char* buffer){
    fpga_program_reset_do(flash_config);
    fpga_spi_reset_cycle(&fpga_env5, buffer[2]);
}


void fpga_toggle_led(void){
    uint8_t data_rx0[3] = {0};
    uint8_t data_toggle_led[3] = {0x08, 0x00, 0x00};
    
    fpga_spi_send_data(&fpga_env5, data_toggle_led, data_rx0);
}


// ======================== CALLABLE FUNCS ==========================
bool apply_rpc_callback(char* buffer, size_t length, bool ready){    
    if(ready){
        switch(buffer[0]){
            case ECHO:                  echo(buffer, length);                       break;
            case RESET:                 system_reset();                             break;
            case GET_SYSTEM_STATE:      get_charac_system();                        break;
            case GET_CHARAC_DAQ:        get_charac_daq();                           break;   
            case ENABLE_LED:            enable_led();                               break;
            case DISABLE_LED:           disable_led();                              break;
            case TOGGLE_LED:            toggle_led();                               break;
            case START_DAQ:             start_daq();                                break;
            case STOP_DAQ:              stop_daq();                                 break;
            case SET_PERIOD_DAQ:        update_period_daq(buffer);                  break;
            case SET_BATCH_DAQ:         set_batch_daq(buffer);                      break;
            /* ADDON FOR FLASH / FPGA INTERACTION */
            case FPGA_INIT_PIN:         init_pin_fpga();                            break;
            case FPGA_POWER_STATE:      set_state_fpga_power(buffer);               break;
            case FPGA_DO_RESET:         fpga_do_reset(buffer);                      break;
            case FLASH_INFOS:           get_flash_infos();                          break;
            case FLASH_START_ERASE_ALL: start_erasing_flash_all();                  break;
            case FLASH_START_ERASE_SEC: start_erasing_flash_sector();               break;
            case FLASH_CHECK_ERASE:     check_erasing_flash();                      break;
            case FLASH_SET_ADDR_UPPER:  set_flash_starting_address_upper(buffer);   break;
            case FLASH_SET_ADDR_LOWER:  set_flash_starting_address_lower(buffer);   break;
            case FLASH_GET_ADDR:        get_flash_starting_address();               break;
            case FLASH_READ_DATA:       read_flash_data();                          break;
            case FLASH_WRITE_BUFFER:    write_data_into_flash_buffer(buffer);       break;
            case FLASH_WRITE_DATA:      write_buffer_into_flash();                  break;
            case FPGA_SEND_DATA:        fpga_toggle_led();                          break;
            default:                    tight_loop_contents();                      break;        
        }  
    }
    return true;
}
      