#ifndef RPC_CALLBACKS_H_
#define RPC_CALLBACKS_H_


#include "pico/stdlib.h"


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



/*! \brief Function for processing the Remote Procedure Calls (RPC) with buffer content from an interface
* \param buffer    Char array with content to handle 
* \param length    Length of the char array
* \param ready     Flag indicating if the buffer is ready to be processed
* \return          True if a valid RPC command was found and processed, false otherwise  
*/
bool apply_rpc_callback(char* buffer, size_t length, bool ready);


#endif
