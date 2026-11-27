import pytest

from api.definitions import BitstreamVendor
from api.src.bin_build import BitstreamReader

PAYLOAD = b"\xff" * 32 + b"\x00\x00\x00\xbb\x11\x22\x00\x44" + b"\xaa\x99\x55\x66" + bytes(range(64))


def _write(tmp_path, name: str, content: bytes):
    path = tmp_path / name
    path.write_bytes(content)
    return path


def test_amd_bit_file_header_is_removed(tmp_path):
    header = b"\x00\x09\x0f\xf0\x0f\xf0\x0f\xf0\x0f\xf0\x00\x01a\x00\x0adesign;UTF\x00e\x00\x00\x00\x60"
    path = _write(tmp_path, "design.bit", header + PAYLOAD)

    assert BitstreamReader.read_amd(path) == PAYLOAD
    assert BitstreamReader.read_amd(path, remove_header=False) == header + PAYLOAD


def test_amd_bin_file_stays_unchanged(tmp_path):
    path = _write(tmp_path, "design.bin", PAYLOAD)

    assert BitstreamReader.read_amd(str(path)) == PAYLOAD


def test_amd_rejects_files_without_padding_or_sync_word(tmp_path):
    with pytest.raises(ValueError, match="padding"):
        BitstreamReader.read_amd(_write(tmp_path, "a.bit", b"\x01" * 100))
    with pytest.raises(ValueError, match="Sync word"):
        BitstreamReader.read_amd(_write(tmp_path, "b.bit", b"\xff" * 32 + b"\x01" * 100))


def test_lattice_accepts_only_bin_files(tmp_path):
    assert (
        BitstreamReader.read_lattice(_write(tmp_path, "design.bin", b"\x7e\xaa\x99\x7e"))
        == b"\x7e\xaa\x99\x7e"
    )
    with pytest.raises(ValueError, match="extension"):
        BitstreamReader.read_lattice(_write(tmp_path, "design.bit", b"\x7e\xaa\x99\x7e"))


def test_cologne_chip_accepts_binary_bit_file_but_not_the_ascii_cfg(tmp_path):
    content = bytes([0xD9, 0x01, 0xED, 0x00])

    assert BitstreamReader.read_cologne_chip(_write(tmp_path, "blink_00.cfg.bit", content)) == content
    with pytest.raises(ValueError, match="ASCII"):
        BitstreamReader.read_cologne_chip(_write(tmp_path, "blink_00.cfg", b"D9 01 ED 00\n"))


def test_empty_files_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="empty"):
        BitstreamReader.read_lattice(_write(tmp_path, "design.bin", b""))


def test_read_accepts_the_vendor_enum(tmp_path):
    path = _write(tmp_path, "design.bin", PAYLOAD)

    assert BitstreamReader.read(BitstreamVendor.AMD, path) == PAYLOAD
    assert BitstreamReader.read(BitstreamVendor.LATTICE, path) == PAYLOAD
    assert (
        BitstreamReader.read(BitstreamVendor.CCGM, _write(tmp_path, "blink2.cfg.bit", b"\xd9\x02"))
        == b"\xd9\x02"
    )
    with pytest.raises(ValueError, match="Unknown FPGA vendor"):
        BitstreamReader.read("intel", path)
