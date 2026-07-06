#ifndef RPC_CALLBACKS_H_
#define RPC_CALLBACKS_H_


#include "pico/stdlib.h"
#include "hal/transport/transport.h"


typedef enum {
    ECHO = 0,
    RESET,
    GET_SYSTEM_STATE,
    GET_NUMBER_DAQ,
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
* \param data       Pointer to the transport RX data buffer
* \return           True if a valid RPC command was found and processed, false otherwise  
*/
bool apply_rpc_callback(transport_rx_buffer_t *data);


#endif
