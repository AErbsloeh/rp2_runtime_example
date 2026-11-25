#ifndef DAQ_READER_H_
#define DAQ_READER_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define DAQ_READER_MAX_STREAMS 16

/*! \brief Error codes of daq_reader_create / daq_reader_start (negative) and daq_reader_stats_t.error (positive: errno) */
#define DAQ_READER_ERR_EOF      (-1)    // connection to the device was closed
#define DAQ_READER_ERR_CONFIG   (-2)    // invalid stream configuration
#define DAQ_READER_ERR_STATE    (-3)    // reader is already started

/*! \brief Push functions of liblsl (lsl_push_sample_itp and lsl_push_chunk_itnp), they are given by the caller, so this
*   library needs neither the header nor the library of liblsl
*/
typedef int32_t (*daq_push_sample_fn)(void *outlet, const int32_t *data, double timestamp, int32_t pushthrough);
typedef int32_t (*daq_push_chunk_fn)(void *outlet, const int32_t *data, unsigned long data_elements,
                                     const double *timestamps, int32_t pushthrough);

/*! \brief Description of one DAQ stream of the device (see the DAQ packet layout of the firmware)
*   Packet: head (1) | index (1) | timestamp [us] (8, batch: first + last = 16) | data (little endian) | CRC16 (2, optional) | tail (1)
* \param head               Head byte of the packets of this stream
* \param tail               Tail byte of the packets of this stream
* \param is_batch           1 if the packet contains num_samples samples (chunk), 0 for one sample
* \param is_signed          1 if the data values are signed
* \param bytes_sample       Number of bytes of one data value (1 ... 4)
* \param has_crc            1 if the packet contains a CRC-16-CCITT (init 0xFFFF) over all bytes before the CRC
* \param num_channels       Number of channels
* \param num_samples        Number of samples of one packet (1 if not batch)
* \param num_bytes_total    Total number of bytes of one packet
* \param outlet             LSL outlet (handle) with num_channels channels in the format int32 receiving the data
*/
typedef struct {
    uint8_t head;
    uint8_t tail;
    uint8_t is_batch;
    uint8_t is_signed;
    uint8_t bytes_sample;
    uint8_t has_crc;
    uint16_t num_channels;
    uint16_t num_samples;
    uint32_t num_bytes_total;
    void *outlet;
} daq_stream_config_t;

typedef struct {
    uint64_t num_packets;           // valid packets
    uint64_t num_lost_packets;      // packets missing in the (wrapping) packet index
    uint64_t num_skipped_bytes;     // bytes dropped for re-synchronisation (invalid head, tail or CRC)
    uint64_t num_push_errors;       // failed pushes into LSL
    int32_t error;                  // 0 or error that stopped the reader (errno or DAQ_READER_ERR_*)
    int32_t running;                // 1 while the reader thread is running
} daq_reader_stats_t;

typedef struct daq_reader daq_reader_t;

/*! \brief Number of bytes of one packet with the given configuration (without checking the outlet)
* \return   Number of bytes or 0 if the configuration is invalid
*/
uint32_t daq_expected_packet_bytes(const daq_stream_config_t *config);

/*! \brief Checksum CRC-16-CCITT (polynomial 0x1021, init 0xFFFF), same as binascii.crc_hqx(data, 0xFFFF)
*/
uint16_t daq_crc16_ccitt(const uint8_t *data, size_t length);

/*! \brief Creating a reader for the given streams
* \param configs        Array with the stream configurations (unique head bytes, size has to match the layout)
* \param num_streams    Number of streams (1 ... DAQ_READER_MAX_STREAMS)
* \param push_sample    Function for pushing one sample into the outlet
* \param push_chunk     Function for pushing a chunk with timestamps into the outlet (needed for batch streams)
* \return               Handle or NULL if the configuration is invalid
*/
daq_reader_t *daq_reader_create(const daq_stream_config_t *configs, size_t num_streams,
                                daq_push_sample_fn push_sample, daq_push_chunk_fn push_chunk);

/*! \brief Starting the reader thread, which reads the file descriptor (serial port or socket), decodes the packets of
*   all streams and pushes the data with the timestamps (in seconds) into the LSL outlets
* \param reader     Handle of the reader
* \param fd         File descriptor of the opened interface to the device (not closed by the reader)
* \return           0 or a negative error code (DAQ_READER_ERR_*) or errno
*/
int daq_reader_start(daq_reader_t *reader, int fd);

/*! \brief Stopping the reader thread (blocks until the thread is finished), the file descriptor stays open */
void daq_reader_stop(daq_reader_t *reader);

/*! \brief Getting the statistics of the reader (can be called while the reader is running) */
void daq_reader_get_stats(const daq_reader_t *reader, daq_reader_stats_t *stats);

/*! \brief Stopping and freeing the reader */
void daq_reader_destroy(daq_reader_t *reader);

#ifdef __cplusplus
}
#endif

#endif
