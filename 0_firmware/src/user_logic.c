#include "src/user_logic.h"


bool run_sequential(const processing_config_t *config, const uint8_t *in, uint8_t *out){
    const size_t num_bytes_in = processing_get_num_bytes(&config->in);
    const size_t num_bytes_out = processing_get_num_bytes(&config->out);

    // Example processing: simply copy input to output, halving the value of each byte
    for(size_t idx = 0; idx < num_bytes_out; idx++){
        out[idx] = (idx < num_bytes_in) ? in[idx] >> 1 : 0;
    }
    return true;
}
