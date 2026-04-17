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

    char buffer_send[19];
    buffer_send[0] = GET_SYSTEM_STATE;
    buffer_send[1] = (uint8_t)(get_system_state() >> 0);
    buffer_send[2] = (uint8_t)(get_system_state() >> 8);
    buffer_send[3] = (uint8_t)(clk_val >> 0);
    buffer_send[4] = (uint8_t)(clk_val >> 8);
    buffer_send[5] = (uint8_t)(get_state_default_led() << 0x00);
    buffer_send[6] = (uint8_t)(get_state_default_led() << 0x08);
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

    char buffer_send[20];
    buffer_send[0] = GET_CHARAC_DAQ;
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


// ======================== CALLABLE FUNCS ==========================
bool apply_rpc_callback(char* buffer, size_t length, bool ready){    
    if(ready){
        switch(buffer[0]){
            case ECHO:                  echo(buffer, length);                   break;
            case RESET:                 system_reset();                         break;
            case GET_SYSTEM_STATE:      get_charac_system();                    break;
            case GET_CHARAC_DAQ:        get_charac_daq();                       break;   
            case ENABLE_LED:            enable_led();                           break;
            case DISABLE_LED:           disable_led();                          break;
            case TOGGLE_LED:            toggle_led();                           break;
            case START_DAQ:             start_daq();                            break;
            case STOP_DAQ:              stop_daq();                             break;
            case SET_PERIOD_DAQ:        update_period_daq(buffer);              break;
            case SET_BATCH_DAQ:         set_batch_daq(buffer);                  break;
            default:                    tight_loop_contents();                  break;        
        }  
    }
    return true;
}
      