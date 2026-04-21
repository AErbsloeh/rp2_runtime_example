#include "src/testbench.h"
#include "hardware_io.h"


// ============================ HELP FUNCTION FOR TESTING ============================
void mode_tb_switch(void){
		sleep_us(1);
}


// ============================ MAIN FUNCTION ============================
bool run_testbench(testbench_mode_t mode){
    set_system_state(STATE_TEST);
    switch(mode){
        case TB_NONE:   sleep_us(1);                                break;
        case TB_MUX:    mode_tb_switch();             			    break;
        default:        set_system_state(STATE_ERROR);              break;
    }
    set_system_state(STATE_IDLE);
    return true;
}
