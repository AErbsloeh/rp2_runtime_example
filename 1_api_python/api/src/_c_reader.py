import ctypes
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from ctypes import POINTER, c_int, c_int32, c_size_t, c_uint8, c_uint16, c_uint32, c_uint64, c_void_p
from dataclasses import dataclass
from logging import getLogger
from pathlib import Path

from ._helper import DataAcquisitionConfig

MAX_STREAMS = 16
ENV_LIBRARY_PATH = "DAQ_READER_LIB"
ERR_EOF = -1
_library: ctypes.CDLL | None = None


class _StreamConfig(ctypes.Structure):
    _fields_ = [
        ("head", c_uint8),
        ("tail", c_uint8),
        ("is_batch", c_uint8),
        ("is_signed", c_uint8),
        ("bytes_sample", c_uint8),
        ("has_crc", c_uint8),
        ("num_channels", c_uint16),
        ("num_samples", c_uint16),
        ("num_bytes_total", c_uint32),
        ("outlet", c_void_p),
    ]


class _Stats(ctypes.Structure):
    _fields_ = [
        ("num_packets", c_uint64),
        ("num_lost_packets", c_uint64),
        ("num_skipped_bytes", c_uint64),
        ("num_push_errors", c_uint64),
        ("error", c_int32),
        ("running", c_int32),
    ]


@dataclass(frozen=True)
class DaqReaderStats:
    """Statistics of the C reader
    Attributes:
        num_packets:        Number of valid packets
        num_lost_packets:   Number of packets missing in the (wrapping) packet index
        num_skipped_bytes:  Number of bytes dropped for re-synchronisation (invalid head, tail or CRC)
        num_push_errors:    Number of failed pushes into LSL
        error:              0 or the error that stopped the reader (errno, -1: connection closed)
        running:            Boolean if the reader thread is running
    """

    num_packets: int
    num_lost_packets: int
    num_skipped_bytes: int
    num_push_errors: int
    error: int
    running: bool


def get_source_folder() -> Path:
    """Folder with the C source of the reader (api/c_reader in the repository)"""
    return Path(__file__).resolve().parents[1] / "c_reader"


def _library_file_name() -> str:
    return "libdaq_reader" + (".dylib" if sys.platform == "darwin" else ".so")


def _build_folders() -> list[Path]:
    """Folders for the compiled library: build folder of c_reader, fallback temp folder (e.g. read-only installation)"""
    return [get_source_folder() / "build", Path(tempfile.gettempdir()) / "rp2_daq_reader"]


def _get_writable_build_folder() -> Path:
    for folder in _build_folders():
        try:
            folder.mkdir(parents=True, exist_ok=True)
        except OSError:
            continue
        if os.access(folder, os.W_OK):
            return folder
    raise PermissionError(
        "No writable folder for building the C reader, set DAQ_READER_LIB to a compiled library"
    )


def find_library() -> Path | None:
    """Searching the compiled library: path in the environment variable DAQ_READER_LIB or the build folders"""
    candidates = []
    if os.environ.get(ENV_LIBRARY_PATH):
        candidates.append(Path(os.environ[ENV_LIBRARY_PATH]))
    candidates += [folder / _library_file_name() for folder in _build_folders()]
    candidates.append(get_source_folder() / _library_file_name())
    return next((path for path in candidates if path.is_file()), None)


def build_library(output: Path | str | None = None) -> Path:
    """Compiling the C reader with the C compiler of the system (CC or cc, gcc, clang)
    :param output:  Path of the library, default: build folder of c_reader (or temp folder if it is not writable)
    :return:        Path of the library
    """
    source = get_source_folder() / "daq_reader.c"
    if not source.is_file():
        raise FileNotFoundError(f"Source of the C reader not found: {source}")
    compiler = os.environ.get("CC") or next((c for c in ("cc", "gcc", "clang") if shutil.which(c)), None)
    if compiler is None:
        raise RuntimeError("No C compiler found, install one or build c_reader with CMake")
    output = Path(output) if output is not None else _get_writable_build_folder() / _library_file_name()
    output.parent.mkdir(parents=True, exist_ok=True)
    shared = "-dynamiclib" if sys.platform == "darwin" else "-shared"
    command = [
        compiler,
        "-std=c11",
        "-O2",
        "-Wall",
        "-fPIC",
        shared,
        "-o",
        str(output),
        str(source),
        "-pthread",
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Building the C reader failed:\n{result.stderr}")
    return output


def _bind(library: ctypes.CDLL) -> ctypes.CDLL:
    library.daq_reader_create.argtypes = [POINTER(_StreamConfig), c_size_t, c_void_p, c_void_p]
    library.daq_reader_create.restype = c_void_p
    library.daq_reader_start.argtypes = [c_void_p, c_int]
    library.daq_reader_start.restype = c_int
    library.daq_reader_stop.argtypes = [c_void_p]
    library.daq_reader_stop.restype = None
    library.daq_reader_get_stats.argtypes = [c_void_p, POINTER(_Stats)]
    library.daq_reader_get_stats.restype = None
    library.daq_reader_destroy.argtypes = [c_void_p]
    library.daq_reader_destroy.restype = None
    return library


def load_library(path: Path | str | None = None, build_if_missing: bool = True) -> ctypes.CDLL:
    """Loading the C reader library (once per process)
    :param path:            Path of the library, default: search with find_library
    :param build_if_missing:True for compiling the library if it was not found
    :return:                Loaded library
    """
    global _library
    if path is not None:
        return _bind(ctypes.CDLL(str(path)))
    if _library is None:
        if os.name != "posix":
            raise RuntimeError("The C reader is only available on POSIX systems (Linux, macOS)")
        found = find_library()
        if found is None:
            if not build_if_missing:
                raise FileNotFoundError(
                    "C reader library not found, build c_reader or set DAQ_READER_LIB"
                )
            found = build_library()
        _library = _bind(ctypes.CDLL(str(found)))
    return _library


class CDaqReader:
    """Reading the DAQ packets of the device (file descriptor of the serial port or socket), decoding them and pushing
    the data into LSL outlets, all in a C thread (without the GIL)"""

    def __init__(
        self,
        configs: Sequence[DataAcquisitionConfig],
        outlets: Sequence[int],
        push_sample,
        push_chunk,
        library: ctypes.CDLL | None = None,
    ) -> None:
        """
        :param configs:     Configurations of the DAQ streams of the device
        :param outlets:     Handles (addresses) of the LSL outlets, one for each stream, format int32
        :param push_sample: C function lsl_push_sample_itp (or compatible) as ctypes function
        :param push_chunk:  C function lsl_push_chunk_itnp (or compatible) as ctypes function
        :param library:     Loaded C reader library, default: load_library()
        """
        self._handle = None
        if len(configs) != len(outlets) or not 1 <= len(configs) <= MAX_STREAMS:
            raise ValueError(f"1 ... {MAX_STREAMS} streams with one outlet for each stream needed")
        self._logger = getLogger(__name__)
        self._lib = library if library is not None else load_library()
        self._push_functions = (push_sample, push_chunk)
        c_configs = (_StreamConfig * len(configs))()
        for idx, (config, outlet) in enumerate(zip(configs, outlets)):
            c_configs[idx] = _StreamConfig(
                head=config.head_cmd,
                tail=config.tail_cmd,
                is_batch=int(config.send_batch),
                is_signed=int(config.is_signed),
                bytes_sample=config.bytes_sample,
                has_crc=int(config.has_crc),
                num_channels=config.num_channels,
                num_samples=config.num_samples,
                num_bytes_total=config.num_bytes_total,
                outlet=outlet,
            )
        self._handle = self._lib.daq_reader_create(
            c_configs,
            len(configs),
            ctypes.cast(push_sample, c_void_p),
            ctypes.cast(push_chunk, c_void_p),
        )
        if not self._handle:
            raise ValueError(
                "Invalid DAQ stream configuration for the C reader (unique head bytes, packet size?)"
            )

    @classmethod
    def from_pylsl(
        cls, configs: Sequence[DataAcquisitionConfig], outlets: Sequence, library=None
    ) -> "CDaqReader":
        """Creating the reader for pylsl outlets (StreamOutlet), which have the format cf_int32"""
        from pylsl.lib import lib

        return cls(
            configs,
            [outlet.obj.value for outlet in outlets],
            lib.lsl_push_sample_itp,
            lib.lsl_push_chunk_itnp,
            library,
        )

    def start(self, fd: int) -> None:
        """Starting the C thread, which reads from the file descriptor (it stays owned by the caller)"""
        status = self._lib.daq_reader_start(self._handle, fd)
        if status != 0:
            raise RuntimeError(f"Starting the C reader failed (code {status})")

    def stop(self) -> None:
        """Stopping the C thread (waits until it is finished)"""
        self._lib.daq_reader_stop(self._handle)

    def stats(self) -> DaqReaderStats:
        values = _Stats()
        self._lib.daq_reader_get_stats(self._handle, ctypes.byref(values))
        return DaqReaderStats(
            num_packets=values.num_packets,
            num_lost_packets=values.num_lost_packets,
            num_skipped_bytes=values.num_skipped_bytes,
            num_push_errors=values.num_push_errors,
            error=values.error,
            running=bool(values.running),
        )

    def close(self) -> None:
        """Stopping and freeing the reader"""
        if self._handle:
            self._lib.daq_reader_destroy(self._handle)
            self._handle = None

    def __del__(self) -> None:
        self.close()
