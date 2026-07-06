#ifndef FPGA_CALLBACKS_H_
#define FPGA_CALLBACKS_H_


#include "pico/stdlib.h"
#include "callbacks/rpc_callbacks.h"


typedef enum {
    FLASH_INIT = USB_CMD_COUNT,
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
    FPGA_INIT,
    FPGA_PROGRAM_STATE,
    FPGA_PROGRAM_CYCLE,
    FPGA_LOGIC_RESET,
    FPGA_TOGGLE_LED,
    FPGA_POWER_STATE,
} fpga_cmd_t;



/*! \brief Function for processing the Remote Procedure Calls (RPC) with buffer content from an interface
* \param data       Pointer to the transport RX data buffer
* \return           True if a valid RPC command was found and processed, false otherwise  
*/
bool apply_fpga_callback(transport_rx_buffer_t *data);


#endif
