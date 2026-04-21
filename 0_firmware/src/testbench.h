#ifndef TESTBENCH_H_
#define TESTBENCH_H_


#include "pico/stdlib.h"


// =============================== DEFINITIONS ===============================
typedef enum{
    TB_NONE = 0,
    TB_MUX
} testbench_mode_t;


// =============================== FUNCTIONS ===============================
/*! \brief Function for choosing a testbench mode
* \param mode   Unsigned integer for selecting the right testmode
* \return       Boolean if test is done
*/
bool run_testbench(testbench_mode_t mode);


#endif
