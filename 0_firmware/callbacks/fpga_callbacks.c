#include "callbacks/fpga_callbacks.h"
#include "hardware_io.h"
#include "version.h"

#include "peri/fpga/flash.h"
#include "peri/fpga/fpga_spi.h"
#include "peri/fpga/fpga_config.h"


// ======================== FLASH / FPGA CMDS ==========================
#define FPGA_EN_POWER_GPIO 23


static flash_fpga_t *flash_config = &flash_env5;
static fpga_spi_t *fpga_config = &fpga_env5;


void flash_init_phase(void){
    bool state = false;
    if(fpga_program_init(flash_config)){
        state = fpga_program_do(flash_config, true);
    }

    char buffer_send[3] = {FLASH_INIT};
    buffer_send[1] = 0x00;
    buffer_send[2] = (uint8_t)(state);
    transport_write(buffer_send, sizeof(buffer_send));
}


void get_flash_infos(void){
    uint8_t manu_id = flash_get_manufacturer_id(flash_config);
    uint16_t device_id = flash_get_device_id(flash_config);
    uint16_t jedec_id = flash_get_jedec_id(flash_config);
    uint16_t status_reg = flash_get_status_register(flash_config);

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
    transport_write(buffer_send, sizeof(buffer_send));
}


void start_erasing_flash_all(void){
    set_system_state(STATE_ERASE_FLASH);
    flash_erasing_all_start(flash_config);
}


void start_erasing_flash_sector(void){
    set_system_state(STATE_ERASE_FLASH);
    flash_erasing_sector_start(flash_config, flash_config->flash_data.address);
}


void check_erasing_flash(void){
    bool state = false;
    if(get_system_state() == STATE_ERASE_FLASH){
        if(flash_erasing_is_done(flash_config)){
            state = true;
            flash_erasing_stop(flash_config);
            set_system_state(STATE_IDLE);
        }
    }

    char buffer_send[3] = {FLASH_CHECK_ERASE};
    buffer_send[1] = 0x00;
    buffer_send[2] = (uint8_t)(state);
    transport_write(buffer_send, sizeof(buffer_send));
}


void set_flash_starting_address_upper(char* buffer){
    uint16_t upper = (buffer[1] << 8) | buffer[2];
    flash_config->flash_data.address = (uint32_t)(upper << 16);
    flash_config->flash_data.position = 0;
}


void set_flash_starting_address_lower(char* buffer){
    uint16_t lower = (buffer[1] << 8) | buffer[2];
    flash_config->flash_data.address |= lower;
    flash_config->flash_data.position = 0;
}


void get_flash_starting_address(void){
    char buffer_send[1 + sizeof(flash_config->flash_data.address)] = {FLASH_GET_ADDR};
    for(size_t idx = 0; idx < sizeof(flash_config->flash_data.address); idx++){
        buffer_send[1 + idx] = (uint8_t)(flash_config->flash_data.address >> (idx * 8));
    }
    transport_write(buffer_send, sizeof(buffer_send));
}


void read_flash_data(void){
    flash_read_data_from_buffer(flash_config);

    char buffer_send[1 + flash_config->flash_data.position_max];
    buffer_send[0] = FLASH_READ_DATA;
    for(size_t idx = 0; idx < flash_config->flash_data.position_max; idx++){
        buffer_send[1 + idx] = (uint8_t)(flash_config->flash_data.data[idx]);
    }
    transport_write(buffer_send, sizeof(buffer_send));
}


void write_data_into_flash_buffer(char* buffer){    
    flash_data_write_byte(&flash_config->flash_data, buffer[1]);
    flash_data_write_byte(&flash_config->flash_data, buffer[2]);
}


void write_buffer_into_flash(void){
    bool state = flash_write_data_from_buffer(flash_config);

    char buffer_send[3] = {FLASH_WRITE_DATA};
    buffer_send[1] = 0x00;
    buffer_send[2] = (uint8_t)(state);
    transport_write(buffer_send, sizeof(buffer_send));
}


void fpga_init_phase(void){
    gpio_init(FPGA_EN_POWER_GPIO);
    gpio_set_dir(FPGA_EN_POWER_GPIO, GPIO_OUT);
    gpio_put(FPGA_EN_POWER_GPIO, true);

    bool state = fpga_spi_init(fpga_config);

    char buffer_send[3] = {FPGA_INIT};
    buffer_send[1] = 0x00;
    buffer_send[2] = (uint8_t)(state);
    transport_write(buffer_send, sizeof(buffer_send));
}


void fpga_set_program_state(char* buffer){
    bool state = (buffer[2] == 0x01);
    fpga_program_do(flash_config, state);
}


void fpga_program_do_cycle(void){
    fpga_program_reset_do(flash_config);
}


void fpga_set_power_state(char* buffer){
    bool state = (buffer[2] == 0x01);
    gpio_put(FPGA_EN_POWER_GPIO, state);
}


void fpga_logic_do_reset(char* buffer){
    fpga_spi_reset_cycle(fpga_config, buffer[2]);
}


void fpga_logic_data_transmission(char* buffer){
    uint8_t data_rx0[3] = {0};
    uint8_t data_toggle_led[3] = {0x08, 0x00, 0x00};
    
    fpga_spi_send_data(fpga_config, data_toggle_led, data_rx0);

    char buffer_send[1 + sizeof(data_rx0)] = {FPGA_LOGIC_RXTX};
    for(size_t idx = 0; idx < sizeof(data_rx0); idx++){
        buffer_send[1 + idx] = (uint8_t)(data_rx0[idx]);
    }
    transport_write(buffer_send, sizeof(buffer_send));
}


// ======================== CALLABLE FUNCS ==========================
bool apply_fpga_callback(char* buffer, size_t length, bool ready){    
    bool valid_state = true;
    if(ready){
        switch(buffer[0]){
            case FLASH_INIT:            flash_init_phase();                         break;
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
            case FPGA_INIT:             fpga_init_phase();                          break;
            case FPGA_POWER_STATE:      fpga_set_power_state(buffer);               break;
            case FPGA_PROGRAM_STATE:    fpga_set_program_state(buffer);             break;
            case FPGA_PROGRAM_CYCLE:    fpga_program_do_cycle();                    break;
            case FPGA_LOGIC_RESET:      fpga_logic_do_reset(buffer);                break;
            case FPGA_LOGIC_RXTX:       fpga_logic_data_transmission(buffer);       break;
            default:                    valid_state = false;                        break;        
        }  
    }
    return valid_state;
}