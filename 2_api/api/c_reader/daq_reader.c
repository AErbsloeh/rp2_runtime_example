#define _POSIX_C_SOURCE 200809L

#include "daq_reader.h"

#include <errno.h>
#include <poll.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define READ_CHUNK_BYTES    4096
#define POLL_TIMEOUT_MS     50

struct daq_reader {
    daq_stream_config_t config[DAQ_READER_MAX_STREAMS];
    size_t num_streams;
    int8_t stream_of_head[256];
    uint8_t last_index[DAQ_READER_MAX_STREAMS];
    uint8_t has_last_index[DAQ_READER_MAX_STREAMS];
    size_t max_packet_bytes;

    daq_push_sample_fn push_sample;
    daq_push_chunk_fn push_chunk;

    uint8_t *buffer;
    size_t buffer_length;
    size_t buffer_capacity;
    int32_t *values;
    double *timestamps;

    pthread_t thread;
    int fd;
    int is_started;
    atomic_int stop_request;
    atomic_int running;
    atomic_int error;
    atomic_uint_fast64_t num_packets;
    atomic_uint_fast64_t num_lost_packets;
    atomic_uint_fast64_t num_skipped_bytes;
    atomic_uint_fast64_t num_push_errors;
};


// ============================== HELPERS ==============================
uint16_t daq_crc16_ccitt(const uint8_t *data, size_t length) {
    uint16_t crc = 0xFFFF;
    for (size_t idx = 0; idx < length; idx++) {
        crc ^= (uint16_t)((uint16_t)data[idx] << 8);
        for (int bit = 0; bit < 8; bit++) {
            crc = (crc & 0x8000) ? (uint16_t)((crc << 1) ^ 0x1021) : (uint16_t)(crc << 1);
        }
    }
    return crc;
}

uint32_t daq_expected_packet_bytes(const daq_stream_config_t *config) {
    if (config->bytes_sample < 1 || config->bytes_sample > 4 || config->num_channels < 1 || config->num_samples < 1) {
        return 0;
    }
    if (!config->is_batch && config->num_samples != 1) {
        return 0;
    }
    uint32_t num_values = (uint32_t)config->num_channels * config->num_samples;
    uint32_t num_bytes_timestamp = config->is_batch ? 16 : 8;
    return 3 + num_bytes_timestamp + num_values * config->bytes_sample + (config->has_crc ? 2 : 0);
}

static uint64_t read_u64_le(const uint8_t *data) {
    uint64_t value = 0;
    for (int idx = 7; idx >= 0; idx--) {
        value = (value << 8) | data[idx];
    }
    return value;
}

static int is_valid_packet(const daq_stream_config_t *config, const uint8_t *packet) {
    size_t length = config->num_bytes_total;
    if (packet[length - 1] != config->tail) {
        return 0;
    }
    if (config->has_crc) {
        uint16_t received = (uint16_t)(packet[length - 3] | (packet[length - 2] << 8));
        return daq_crc16_ccitt(packet, length - 3) == received;
    }
    return 1;
}


// ============================== PACKET PROCESSING ==============================
static void handle_packet(daq_reader_t *reader, size_t stream, const uint8_t *packet) {
    const daq_stream_config_t *config = &reader->config[stream];
    uint8_t index = packet[1];

    if (reader->has_last_index[stream]) {
        uint8_t num_lost = (uint8_t)(index - reader->last_index[stream] - 1);
        if (num_lost) {
            atomic_fetch_add(&reader->num_lost_packets, num_lost);
        }
    }
    reader->last_index[stream] = index;
    reader->has_last_index[stream] = 1;

    size_t num_values = (size_t)config->num_channels * config->num_samples;
    const uint8_t *data = packet + 2 + (config->is_batch ? 16 : 8);
    for (size_t idx = 0; idx < num_values; idx++) {
        uint32_t raw = 0;
        for (int byte = config->bytes_sample - 1; byte >= 0; byte--) {
            raw = (raw << 8) | data[idx * config->bytes_sample + byte];
        }
        if (config->is_signed && config->bytes_sample < 4 && (raw & (1u << (8 * config->bytes_sample - 1)))) {
            raw |= ~((1u << (8 * config->bytes_sample)) - 1u);
        }
        reader->values[idx] = (int32_t)raw;
    }

    int32_t status;
    if (config->is_batch) {
        double t_first = (double)read_u64_le(packet + 2);
        double t_last = (double)read_u64_le(packet + 10);
        double dt = (t_last - t_first) / (double)(config->num_samples > 1 ? config->num_samples - 1 : 1);
        for (size_t idx = 0; idx < config->num_samples; idx++) {
            reader->timestamps[idx] = 1e-6 * (t_first + dt * (double)idx);
        }
        status = reader->push_chunk(config->outlet, reader->values, (unsigned long)num_values, reader->timestamps, 1);
    } else {
        status = reader->push_sample(config->outlet, reader->values, 1e-6 * (double)read_u64_le(packet + 2), 1);
    }
    if (status < 0) {
        atomic_fetch_add(&reader->num_push_errors, 1);
    }
    atomic_fetch_add(&reader->num_packets, 1);
}

/* Decoding all complete packets in the buffer. Packets of different streams can arrive in any order. The head byte
 * selects the stream (and thus the length), tail byte and CRC validate the candidate. On a mismatch only one byte is
 * dropped, so the parser re-synchronises on the next real packet start. */
static void process_buffer(daq_reader_t *reader) {
    size_t position = 0;
    while (position < reader->buffer_length) {
        int stream = reader->stream_of_head[reader->buffer[position]];
        if (stream < 0) {
            position++;
            atomic_fetch_add(&reader->num_skipped_bytes, 1);
            continue;
        }
        const daq_stream_config_t *config = &reader->config[stream];
        if (reader->buffer_length - position < config->num_bytes_total) {
            break;
        }
        if (is_valid_packet(config, reader->buffer + position)) {
            handle_packet(reader, (size_t)stream, reader->buffer + position);
            position += config->num_bytes_total;
        } else {
            position++;
            atomic_fetch_add(&reader->num_skipped_bytes, 1);
        }
    }
    if (position > 0) {
        memmove(reader->buffer, reader->buffer + position, reader->buffer_length - position);
        reader->buffer_length -= position;
    }
}

static void *reader_thread(void *argument) {
    daq_reader_t *reader = (daq_reader_t *)argument;
    struct pollfd poll_fd = {.fd = reader->fd, .events = POLLIN, .revents = 0};

    while (!atomic_load(&reader->stop_request)) {
        int status = poll(&poll_fd, 1, POLL_TIMEOUT_MS);
        if (status < 0) {
            if (errno == EINTR) {
                continue;
            }
            atomic_store(&reader->error, errno);
            break;
        }
        if (status == 0) {
            continue;
        }
        if (!(poll_fd.revents & POLLIN)) {
            atomic_store(&reader->error, (poll_fd.revents & POLLNVAL) ? EBADF : EIO);
            break;
        }
        ssize_t num_read = read(reader->fd, reader->buffer + reader->buffer_length,
                                reader->buffer_capacity - reader->buffer_length);
        if (num_read < 0) {
            if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK) {
                continue;
            }
            atomic_store(&reader->error, errno);
            break;
        }
        if (num_read == 0) {
            atomic_store(&reader->error, DAQ_READER_ERR_EOF);
            break;
        }
        reader->buffer_length += (size_t)num_read;
        process_buffer(reader);
    }
    atomic_store(&reader->running, 0);
    return NULL;
}


// ============================== PUBLIC FUNCTIONS ==============================
daq_reader_t *daq_reader_create(const daq_stream_config_t *configs, size_t num_streams,
                                daq_push_sample_fn push_sample, daq_push_chunk_fn push_chunk) {
    if (configs == NULL || num_streams < 1 || num_streams > DAQ_READER_MAX_STREAMS || push_sample == NULL) {
        return NULL;
    }
    daq_reader_t *reader = calloc(1, sizeof(daq_reader_t));
    if (reader == NULL) {
        return NULL;
    }
    memset(reader->stream_of_head, -1, sizeof(reader->stream_of_head));

    size_t max_values = 0;
    for (size_t idx = 0; idx < num_streams; idx++) {
        const daq_stream_config_t *config = &configs[idx];
        uint32_t expected_bytes = daq_expected_packet_bytes(config);
        if (expected_bytes == 0 || expected_bytes != config->num_bytes_total || config->outlet == NULL ||
            reader->stream_of_head[config->head] >= 0 || (config->is_batch && push_chunk == NULL)) {
            free(reader);
            return NULL;
        }
        reader->config[idx] = *config;
        reader->stream_of_head[config->head] = (int8_t)idx;
        if (config->num_bytes_total > reader->max_packet_bytes) {
            reader->max_packet_bytes = config->num_bytes_total;
        }
        size_t num_values = (size_t)config->num_channels * config->num_samples;
        if (num_values > max_values) {
            max_values = num_values;
        }
    }
    reader->num_streams = num_streams;
    reader->push_sample = push_sample;
    reader->push_chunk = push_chunk;

    /* Incomplete packets stay in the buffer (shorter than the longest packet), the rest is free for reading */
    reader->buffer_capacity = reader->max_packet_bytes + READ_CHUNK_BYTES;
    reader->buffer = malloc(reader->buffer_capacity);
    reader->values = malloc(max_values * sizeof(int32_t));
    reader->timestamps = malloc(max_values * sizeof(double));
    if (reader->buffer == NULL || reader->values == NULL || reader->timestamps == NULL) {
        daq_reader_destroy(reader);
        return NULL;
    }
    return reader;
}

int daq_reader_start(daq_reader_t *reader, int fd) {
    if (reader == NULL || fd < 0) {
        return DAQ_READER_ERR_CONFIG;
    }
    if (reader->is_started) {
        return DAQ_READER_ERR_STATE;
    }
    reader->fd = fd;
    reader->buffer_length = 0;
    memset(reader->has_last_index, 0, sizeof(reader->has_last_index));
    atomic_store(&reader->stop_request, 0);
    atomic_store(&reader->error, 0);
    atomic_store(&reader->num_packets, 0);
    atomic_store(&reader->num_lost_packets, 0);
    atomic_store(&reader->num_skipped_bytes, 0);
    atomic_store(&reader->num_push_errors, 0);
    atomic_store(&reader->running, 1);

    int status = pthread_create(&reader->thread, NULL, reader_thread, reader);
    if (status != 0) {
        atomic_store(&reader->running, 0);
        return status;
    }
    reader->is_started = 1;
    return 0;
}

void daq_reader_stop(daq_reader_t *reader) {
    if (reader == NULL || !reader->is_started) {
        return;
    }
    atomic_store(&reader->stop_request, 1);
    pthread_join(reader->thread, NULL);
    reader->is_started = 0;
}

void daq_reader_get_stats(const daq_reader_t *reader, daq_reader_stats_t *stats) {
    if (reader == NULL || stats == NULL) {
        return;
    }
    stats->num_packets = atomic_load(&reader->num_packets);
    stats->num_lost_packets = atomic_load(&reader->num_lost_packets);
    stats->num_skipped_bytes = atomic_load(&reader->num_skipped_bytes);
    stats->num_push_errors = atomic_load(&reader->num_push_errors);
    stats->error = atomic_load(&reader->error);
    stats->running = atomic_load(&reader->running);
}

void daq_reader_destroy(daq_reader_t *reader) {
    if (reader == NULL) {
        return;
    }
    daq_reader_stop(reader);
    free(reader->buffer);
    free(reader->values);
    free(reader->timestamps);
    free(reader);
}
