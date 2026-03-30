#include "callbacks/rpc_callbacks.h"
#include "hardware_io.h"
#include "version.h"


// ============================= COMMANDS =============================
typedef enum {
    ECHO = 0,
    RESET,
    CLOCK_SYS,
    STATE_SYS,
    STATE_PIN,
    RUNTIME,
    FIRMWARE,
	TEMP_MCU,
    ENABLE_LED,
    DISABLE_LED,
    TOGGLE_LED,
    START_SAMPLE_DAQ,
    START_BATCH_DAQ,
    STOP_DAQ,
    UPDATE_DAQ,
    ASK_BATCH_DAQ
} usb_cmd_t;


// ========================== PROCOTOL FUNCS ==========================
void echo(char* buffer, size_t length){
    usb_send_bytes(buffer, length);
}


void system_reset(void){
    reset_pico_mcu(true);
}


void get_state_system(void){
    char buffer_send[3];
    buffer_send[0] = STATE_SYS;
    buffer_send[1] = 0x00;
    buffer_send[2] = system_state;

    usb_send_bytes(buffer_send, sizeof(buffer_send));
}


void get_clock_system(void){
    uint16_t clk_val = (uint16_t)(clock_get_hz(clk_sys) / 10000);

    char buffer_send[3];
    buffer_send[0] = CLOCK_SYS;
    buffer_send[1] = (uint8_t)(clk_val >> 0);
    buffer_send[2] = (uint8_t)(clk_val >> 8);
    usb_send_bytes(buffer_send, sizeof(buffer_send));
}


void get_state_pin(void){
    char buffer_send[3];
    buffer_send[0] = STATE_PIN;
    buffer_send[1] = 0x02;
    buffer_send[2] = (get_state_default_led() << 0x00);
    usb_send_bytes(buffer_send, sizeof(buffer_send));
}


void get_runtime(void){
    char buffer_send[9];
    buffer_send[0] = RUNTIME;
    uint64_t runtime = get_runtime_ms();
    for(uint8_t idx = 0; idx < 8; idx++){
        buffer_send[idx+1] = (uint8_t)runtime;
        runtime >>= 8;
    }
    usb_send_bytes(buffer_send, sizeof(buffer_send));
}


void get_firmware_version(void){
    char buffer_send[3];
    buffer_send[0] = FIRMWARE;
    buffer_send[1] = PROGRAM_VERSION[0]-48;
    buffer_send[2] = PROGRAM_VERSION[2]-48;
    usb_send_bytes(buffer_send, sizeof(buffer_send));
}


void get_temp_mcu(void){
    uint16_t temp_raw = rp2_adc_read_raw(&adc_temp);

    char buffer_send[3];
    buffer_send[0] = TEMP_MCU;
    buffer_send[1] = (uint8_t)(temp_raw >> 0);
    buffer_send[2] = (uint8_t)(temp_raw >> 8);
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


void start_sample_daq(void){
    daq_sample_data.send_batch = false;
    daq_start_sampling(&tmr_daq0_hndl);
    set_system_state(STATE_DAQ);
}


void start_batch_daq(void){
    daq_sample_data.send_batch = true;
    daq_start_sampling(&tmr_daq0_hndl);
    set_system_state(STATE_DAQ);
}


void stop_daq(void){
    set_system_state(STATE_IDLE);
    daq_stop_sampling(&tmr_daq0_hndl);
}


void update_daq(char* buffer){
    float new_sampling_rate_hz = (buffer[1] << 8) | (buffer[2] << 0);
    int64_t new_rate_us = (float)-1000000 / new_sampling_rate_hz;
    daq_update_sampling_rate(&tmr_daq0_hndl, new_rate_us);
}


void ask_batch_daq(void){
    char buffer_send[3];
    buffer_send[0] = ASK_BATCH_DAQ;
    buffer_send[1] = 0x00;
    buffer_send[2] = daq_sample_data.num_samples;
    usb_send_bytes(buffer_send, sizeof(buffer_send));
}


// ======================== CALLABLE FUNCS ==========================
bool apply_rpc_callback(char* buffer, size_t length, bool ready){    
    if(ready){
        switch(buffer[0]){
            case ECHO:              echo(buffer, length);                   break;
            case RESET:             system_reset();                         break;
            case CLOCK_SYS:         get_clock_system();                     break;
            case STATE_SYS:         get_state_system();                     break;
            case STATE_PIN:         get_state_pin();                        break; 
            case RUNTIME:           get_runtime();                          break;
			case TEMP_MCU:		    get_temp_mcu();							break;
            case FIRMWARE:          get_firmware_version();                 break;
            case ENABLE_LED:        enable_led();                           break;
            case DISABLE_LED:       disable_led();                          break;
            case TOGGLE_LED:        toggle_led();                           break;
            case START_SAMPLE_DAQ:  start_sample_daq();                     break;
            case START_BATCH_DAQ:   start_batch_daq();                      break;
            case STOP_DAQ:          stop_daq();                             break;
            case UPDATE_DAQ:        update_daq(buffer);                     break;
            case ASK_BATCH_DAQ:     ask_batch_daq();                        break;
            default:                set_system_state(STATE_ERROR);          break;        
        }  
    }
    return true;
}
      