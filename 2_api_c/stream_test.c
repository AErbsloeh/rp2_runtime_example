
#define _DEFAULT_SOURCE  
#define _GNU_SOURCE  

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <fcntl.h>
#include <termios.h>
#include <string.h>
#include <errno.h>

#define BAUD_RATE B115200
#define DEVICE    "/dev/ttyACM0"
#define BUFFER_SIZE 444

static struct termios tty;

int open_serial(const char *path) {
    int fd = open(path, O_RDONLY | O_NOCTTY);
    if (fd < 0) { perror("open"); return -1; }

    if (tcgetattr(fd, &tty) != 0) {
        perror("tcgetattr");
        close(fd);
        return -1;
    }

    // activate raw mode, this is importnatn 
    cfmakeraw(&tty);

    // Set BAUD SR
    cfsetispeed(&tty, BAUD_RATE);
    cfsetospeed(&tty, BAUD_RATE);
    // CS0 = (8 bits per Byte)  | CLOCAL Ignore control lines ||  CREAD READ MODE 
    tty.c_cflag |= CS8 | CLOCAL | CREAD;
    // disable Hardware (RTS/CTS) flow control.
    tty.c_cflag &= ~CRTSCTS;
    // disable  XON/XOFF
    tty.c_iflag &= ~(IXON | IXOFF | IXANY);

    // 255 Bytes
    tty.c_cc[VMIN]  = 255;
    tty.c_cc[VTIME] = 0;

    tcflush(fd, TCIFLUSH);
    if (tcsetattr(fd, TCSANOW, &tty) != 0) {
        perror("tcsetattr");
        close(fd);
        return -1;
    }

    return fd;
}

void read_usb_data(int fd) {
    uint8_t buffer[BUFFER_SIZE];
    while (1) {
        size_t got = 0;
        while (got < sizeof(buffer)) {
            ssize_t n = read(fd, buffer + got, sizeof(buffer) - got);
            if (n < 0) {
                perror("read");
                return;
            }
            got += n;
        }
        for (size_t i = 0; i < 2; i++)
            printf("%02x ", buffer[i]);
        printf("%02x", buffer[BUFFER_SIZE-4]);
        printf("%02x", buffer[BUFFER_SIZE-3]);
        printf("%02x", buffer[BUFFER_SIZE-2]);
        printf("%02x", buffer[BUFFER_SIZE-1]);
        printf("\n");
    }
}

int main(void) {
    int fd = open_serial(DEVICE);
    if (fd < 0) return EXIT_FAILURE;
    read_usb_data(fd);
    return EXIT_SUCCESS;
}
