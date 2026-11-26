#ifndef USER_LOGIC_H_
#define USER_LOGIC_H_


#include "hal/daq/processing.h"


/*! \brief User logic: processes one data block on core 1 (template: echo of the input data into the output data, replace it by your user logic)
* \param config     Pointer to the settings (layout of the input and output block)
* \param in         Pointer to the input block (processing_get_num_bytes(&config->in) bytes), do not modify
* \param out        Pointer to the buffer for the output block (processing_get_num_bytes(&config->out) bytes)
* \return           True if the result in out is valid and shall be passed on to core 0, false for no result
*/
bool run_sequential(const processing_config_t *config, const uint8_t *in, uint8_t *out);


#endif
