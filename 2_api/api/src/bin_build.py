from pathlib import Path

from ..definitions import BitstreamVendor


class BitstreamReader:
    """Reading the content of bitstream files of different FPGA vendors as bytes, e.g. for writing them into the flash of the FPGA"""

    AMD_PADDING = b"\xff" * 12
    AMD_SYNC_WORD = b"\xaa\x99\x55\x66"

    @staticmethod
    def _read_file(path_to_bitstream_file: Path | str, suffixes: tuple[str, ...], hint: str) -> bytes:
        path = Path(path_to_bitstream_file)
        if path.suffix.lower() not in suffixes:
            raise ValueError(
                f"Bitstream file must have the extension {' or '.join(suffixes)} ({hint}), got '{path.name}'"
            )
        data = path.read_bytes()
        if len(data) == 0:
            raise ValueError(f"Bitstream file '{path.name}' is empty")
        return data

    @classmethod
    def read_amd(cls, path_to_bitstream_file: Path | str, remove_header: bool = True) -> bytes:
        """Reading the content of a bitstream file (*.bit, *.bin) compiled from AMD Vivado for AMD FPGAs
        :param path_to_bitstream_file:      Path to the bitstream file (*.bin without header needs `write_bitstream -bin_file`)
        :param remove_header:               Boolean for removing the header information of the *.bit file (everything before the padding words)
        :return:                            Byte array with flash content
        """
        data = cls._read_file(path_to_bitstream_file, (".bin", ".bit"), "AMD Vivado")
        if not remove_header:
            return data

        pos_padding = data.find(cls.AMD_PADDING)
        if pos_padding == -1:
            raise ValueError("No padding words found, it is not a bitstream of an AMD FPGA")
        if data.find(cls.AMD_SYNC_WORD, pos_padding) == -1:
            raise ValueError("Sync word not found, it is not a bitstream of an AMD FPGA")
        return data[pos_padding:]

    @classmethod
    def read_lattice(cls, path_to_bitstream_file: Path | str) -> bytes:
        """Reading the content of a bitstream file (*.bin) compiled for Lattice FPGAs
        :param path_to_bitstream_file:      Path to the bitstream file
        :return:                            Byte array with flash content
        """
        return cls._read_file(path_to_bitstream_file, (".bin",), "binary format of Lattice")

    @classmethod
    def read_cologne_chip(cls, path_to_bitstream_file: Path | str) -> bytes:
        """Reading the content of a bitstream file (*.bit, e.g. design_00.cfg.bit) compiled for Cologne Chip GateMate FPGAs
        The binary file is created by `gmpack --bit`, the ASCII file (*.cfg) is not a valid content for the flash
        :param path_to_bitstream_file:      Path to the bitstream file
        :return:                            Byte array with flash content
        """
        return cls._read_file(
            path_to_bitstream_file, (".bit",), "binary format of GateMate, not the ASCII *.cfg"
        )

    @classmethod
    def read(cls, vendor: BitstreamVendor, path_to_bitstream_file: Path | str, **kwargs) -> bytes:
        """Reading the content of a bitstream file of the selected FPGA vendor
        :param vendor:                      BitstreamVendor or its name as string: "amd", "lattice" or "ccgm" ("cologne_chip")
        :param path_to_bitstream_file:      Path to the bitstream file
        :param kwargs:                      Further options of the reader of the vendor (e.g. remove_header for AMD)
        :return:                            Byte array with flash content
        """
        readers = {
            BitstreamVendor.AMD: cls.read_amd,
            BitstreamVendor.LATTICE: cls.read_lattice,
            BitstreamVendor.CCGM: cls.read_cologne_chip,
        }
        if vendor not in readers:
            raise ValueError(f"Unknown FPGA vendor '{vendor}', use one of {list(BitstreamVendor)}")
        return readers[vendor](path_to_bitstream_file, **kwargs)
