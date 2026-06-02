from socket import (
    AF_INET,
    SOCK_STREAM,
    socket,
)
from threading import Thread

from api.src._interface_wifi import InterfaceWifi


def _start_tcp_server(response: bytes) -> tuple[int, list[bytes], Thread]:
    received = []
    server = socket(AF_INET, SOCK_STREAM)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = server.getsockname()[1]

    def serve() -> None:
        client, _ = server.accept()
        with client:
            received.append(client.recv(1024))
            client.sendall(response[:2])
            client.sendall(response[2:])
        server.close()

    thread = Thread(target=serve)
    thread.start()
    return port, received, thread


def test_convert_matches_rpc_command_layout():
    dut = InterfaceWifi(host="127.0.0.1", port=4242)
    assert dut.convert(0x02, 0x1234) == bytes([0x34, 0x12, 0x02])


def test_write_wfb_reads_exact_response_size_from_tcp_stream():
    port, received, thread = _start_tcp_server(b"\x02\x03\x04\x05\x06")
    dut = InterfaceWifi(host="127.0.0.1", port=port, timeout=1.)

    dut.open()
    try:
        response = dut.write_wfb(bytes([0x00, 0x00, 0x02]), size=5)
    finally:
        dut.close()
        thread.join(timeout=1.)

    assert response == bytes([0x02, 0x03, 0x04, 0x05, 0x06])
    assert received == [bytes([0x00, 0x00, 0x02])]
