#ifndef TRANSPORT_H_
#define TRANSPORT_H_

#include "stddef.h"
#include "pico/stdlib.h"

typedef struct {
    bool ready;
    uint8_t length;
    uint8_t position;
    char *data;
} transport_rx_buffer_t;

bool transport_init(transport_rx_buffer_t *rx_buffer);

bool transport_wait_until_connected(void);

void transport_poll_rx(transport_rx_buffer_t *rx_buffer);

size_t transport_write(char *data, size_t len);

#endif