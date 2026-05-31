#include "src/transport.h"
#include "hal/usb/usb.h"

static usb_rp2_t usb_rx_buffer;

static void sync_from_usb(transport_rx_buffer_t *rx_buffer) {
    rx_buffer->ready = usb_rx_buffer.ready;
    rx_buffer->length = usb_rx_buffer.length;
    rx_buffer->position = usb_rx_buffer.position;
    rx_buffer->data = usb_rx_buffer.data;
}

bool transport_init(transport_rx_buffer_t *rx_buffer)
{   
    usb_rx_buffer.ready = rx_buffer->ready;
    usb_rx_buffer.length = rx_buffer->length;
    usb_rx_buffer.position = rx_buffer->position;
    usb_rx_buffer.data = rx_buffer->data;

    bool ok = usb_init(&usb_rx_buffer);
    sync_from_usb(rx_buffer);
    return ok;
}

bool transport_wait_until_connected(void)
{
    return usb_wait_until_connected();
}

void transport_poll_rx(transport_rx_buffer_t *rx_buffer)
{
    usb_handling_fifo_buffer(&usb_rx_buffer);
    sync_from_usb(rx_buffer);
}

size_t transport_write(char *data, size_t len)
{
    return usb_send_bytes(data, len);
}