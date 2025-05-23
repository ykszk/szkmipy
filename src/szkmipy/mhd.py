"""
Read/Write Meta Image
"""

from typing import Dict, Tuple, Union, Any, List, Iterator
import re
from pathlib import Path
import zlib
import os
import copy
import warnings
from collections import OrderedDict
import deflate
import zstandard
from mmap import mmap

import numpy as np

_METATYPE2DTYPE_TABLE = {
    "MET_CHAR": "i1",
    "MET_UCHAR": "u1",
    "MET_SHORT": "i2",
    "MET_USHORT": "u2",
    "MET_INT": "i4",
    "MET_UINT": "u4",
    "MET_LONG": "i8",
    "MET_ULONG": "u8",
    "MET_FLOAT": "f4",
    "MET_DOUBLE": "f8",
}
_DTYPE2METATYPE_TABLE = {
    "int8": "MET_CHAR",
    "uint8": "MET_UCHAR",
    "int16": "MET_SHORT",
    "uint16": "MET_USHORT",
    "int32": "MET_INT",
    "uint32": "MET_UINT",
    "int64": "MET_LONG",
    "uint64": "MET_ULONG",
    "float32": "MET_FLOAT",
    "float64": "MET_DOUBLE",
}


def _str2bool(s: str):
    if s == "True":
        return True
    elif s == "False":
        return False
    raise ValueError("Non boolean string")


def _str2array(string: str):
    for t in [int, float, _str2bool]:
        try:
            l = [t(e) for e in string.split()]
            if len(l) > 1:
                return l
            else:
                return l[0]
        except:
            continue
    return string


def _array2str(array):
    if isinstance(array, str):
        return array
    if hasattr(array, "__iter__"):
        return " ".join([str(e) for e in array])
    else:
        return str(array)


def read_header(filename: Union[Path, str], encoding: str = "ascii") -> Dict[str, Any]:
    """Read meta image header.

    :param str filename: Image filename with extension mhd or mha.
    :return: meta data dictionary.
    :rtype: dict
    """
    filename = str(filename)
    header = OrderedDict()
    with open(filename, "rb") as f:
        meta_regex = re.compile("(.+) = (.*)")
        for line in f:
            line = line.decode(encoding)
            if line == "\n":
                continue  # skip empty line
            match = meta_regex.match(line)
            if match:
                header[match.group(1)] = match.group(2).rstrip()
                if match.group(1) == "ElementDataFile":
                    break
            else:
                raise RuntimeError("Bad meta header line : " + line)
    header = OrderedDict(
        [(key, _str2array(value)) for (key, value) in header.items()]
    )  # convert string into array if possible
    return header


def _get_dim(header: Dict[str, Any]):
    """
    return dim in xyz order
    """
    dim = header["DimSize"]
    if "ElementNumberOfChannels" in header:
        dim = [header["ElementNumberOfChannels"]] + dim
    if not hasattr(dim, "__len__"):
        dim = [dim]
    return dim


def read_memmap(
    filename: Union[Path, str], encoding="ascii"
) -> Tuple[np.memmap, Dict[str, Any]]:
    """Read Meta Image as a memory-map.

    :param str filename: Image filename with extension mhd or mha.
    :return: ND image and meta data.
    :rtype: (numpy.memmap, dict)
    :raises: RuntimeError if image data is compressed
    """
    filename = str(filename)
    header = read_header(filename, encoding)
    data_is_compressed = "CompressedData" in header and header["CompressedData"]
    if data_is_compressed:
        raise RuntimeError("Memory-map cannot be created for compressed data.")
    dtype = np.dtype(_METATYPE2DTYPE_TABLE[header["ElementType"]])
    data_filename = header["ElementDataFile"]
    if data_filename == "LOCAL":  # mha
        numel = np.prod(_get_dim(header))
        data_size = numel * dtype.itemsize
        offset = int(os.path.getsize(filename) - data_size)
        data_filename = filename
    else:
        offset = 0
        if not os.path.isabs(data_filename):  # data_filename is relative
            data_filename = os.path.join(os.path.dirname(filename), data_filename)
    dim = _get_dim(header)
    return (
        np.memmap(
            data_filename, dtype=dtype, mode="r", shape=tuple(dim[::-1]), offset=offset
        ),
        header,
    )


def read(
    filename: Union[Path, str], encoding="ascii"
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Read Meta Image.

    :param str filename: Image filename with extension mhd or mha.
    :return: ND image and meta data.
    :rtype: (numpy.ndarray, dict)

    Examples:
        >>> import mhd
        >>> image, header = mhd.read('filename.mhd')
    """
    filename = str(filename)
    header = read_header(filename, encoding)
    data_is_compressed = "CompressedData" in header and header["CompressedData"]
    data_filename = header["ElementDataFile"]
    if header["ObjectType"] != "Image":
        raise ValueError('ObjectType not "Image" is not supported (yet.)')
    if data_filename == "LIST":
        raise ValueError('ElementDataFile "LIST" is not supported (yet.)')
    if data_filename == "LOCAL":  # mha
        data_filename = filename
        if data_is_compressed:
            data_size = header["CompressedDataSize"]
        else:
            numel = np.prod(_get_dim(header))
            data_size = int(numel) * int(
                np.dtype(_METATYPE2DTYPE_TABLE[header["ElementType"]]).itemsize
            )
        seek_size = os.path.getsize(filename) - data_size
    else:  # mhd
        if not os.path.isabs(data_filename):  # data_filename is relative
            data_filename = os.path.join(os.path.dirname(filename), data_filename)
        data_size = os.path.getsize(data_filename)
        seek_size = 0
    if data_is_compressed:
        with open(data_filename, "rb", buffering=0) as f:
            f.seek(seek_size)
            data = f.read()
        numel = int(np.prod(np.array(_get_dim(header))))
        decompressed_size = (
            numel * np.dtype(_METATYPE2DTYPE_TABLE[header["ElementType"]]).itemsize
        )
        compression_type = header.get("CompressionType", "zlib")
        if compression_type == "zlib":
            data = deflate.zlib_decompress(data, decompressed_size)
        elif compression_type == "zstandard":
            ba = bytearray(decompressed_size)
            dctx = zstandard.ZstdDecompressor()
            with dctx.stream_reader(data) as reader:
                reader.readinto(ba)
            data = ba
        else:
            raise ValueError("Unknown compression type: " + compression_type)
    else:
        with open(data_filename, "rb", buffering=0) as f:
            # Using f.readinto(bytearray) to read binary file into a mutable buffer
            # This is slightly slower than simply doing `data = f.read()`, in which data is bytes and thus immutable
            data = bytearray(data_size)
            f.seek(seek_size)
            f.readinto(data)

    data = np.frombuffer(
        data, dtype=np.dtype(_METATYPE2DTYPE_TABLE[header["ElementType"]])
    )
    dim = _get_dim(header)
    image = np.reshape(data, list(reversed(dim)), order="C")
    try:
        image.setflags(write=True)
    except Exception:
        pass
    return image, header


_no_compression_types = set(
    ["float32", "float64"]
)  # it takes longer time to compress these types


def _is_compression_preferable(np_dtype):
    return not (np_dtype in _no_compression_types)


def create_header(
    shape,
    compress=None,
    last_dim_is_channel=False,
    compression_type: str = None,
):
    """Create Meta Image header.

    :param numpy.ndarray image: Image to be written.
    :param bool compress: (optional) Compress data or not.
    :param bool last_dim_is_channel: (optional) If True, the last dimension is considered as channel dimension.
    :return: meta data dictionary.
    :rtype: dict
    """
    header = {}
    header["ObjectType"] = "Image"
    header["CompressedData"] = str(compress)
    header['NDims'] = len(shape)
    if last_dim_is_channel:
        header["ElementNumberOfChannels"] = shape[-1]
        header["DimSize"] = list(reversed(shape[:-1]))
        header["NDims"] -= 1
    else:
        header["DimSize"] = list(reversed(shape))
    if compress and compression_type is not None:
        if compression_type not in ["zlib", "zstandard"]:
            raise ValueError(
                "Unknown compression type: {0}. Supported types are zlib and zstandard.".format(
                    compression_type
                )
            )
        header["CompressionType"] = compression_type
    return header

import packaging.version
if packaging.version.Version(np.__version__) >= packaging.version.Version("1.24"):
    np_bool8 = bool
else:
    np_bool8 = np.bool_

def write(
    filename: Union[Path, str],
    image: np.ndarray,
    header: Dict[str, Any] = {},
    compression_level=6,
):
    """Write Meta Image.

    :param str filename: Image filename with extension mhd or mha.
    :param numpy.ndarray image: Image to be written.
    :param dict [header]: (optional) Meta data for the image.

    Examples:
        >>> from szkmipy import mhd
        >>> mhd.write('filename.mhd', nparray)
        >>> mhd.write('filename.mhd', nparray, {'CompressedData': True}) # compress output
        >>> mhd.write('filename.mhd', nparray, {'ElementNumberOfChannels': nparray.shape[-1]}) # multiple channels
    """
    filename = str(filename)
    if image.dtype == np_bool8:
        image = image.astype(np.uint8)
    header = copy.deepcopy(header)  # copy given header because this function mutate it
    # Construct header
    h = OrderedDict()
    h["ObjectType"] = "Image"
    # Set image dependent meta data
    h["NDims"] = image.ndim
    h["ElementType"] = _DTYPE2METATYPE_TABLE[image.dtype.name]
    h["CompressedData"] = (
        header["CompressedData"]
        if "CompressedData" in header.keys()
        else _is_compression_preferable(image.dtype.name)
    )
    # Remove redundant keys from given header
    for key in h.keys():
        header.pop(key, None)
    # Merge default and given headers
    h.update(header)
    if ("ElementNumberOfChannels") in h:
        h["ElementNumberOfChannels"] = image.shape[-1]
        h["DimSize"] = reversed(image.shape[:-1])
        h["NDims"] -= 1
    else:
        h["DimSize"] = reversed(image.shape)

    h = OrderedDict(
        [(key, _array2str(value)) for (key, value) in h.items()]
    )  # convert array into string if possible
    filename_base, file_extension = os.path.splitext(os.path.basename(filename))
    compress_data = h["CompressedData"] == "True"  # boolean variable for convenience
    if file_extension == ".mhd":
        if compress_data:
            data_filename = filename_base + ".zraw"
        else:
            data_filename = filename_base + ".raw"
    else:
        if file_extension != ".mha":
            warnings.warn(
                'Unknown file extension "{0}". Saving as a .mha file.'.format(
                    file_extension
                ),
                stacklevel=2,
            )
        data_filename = "LOCAL"
    data = np.ascontiguousarray(image).data
    if compress_data:
        compression_type = header.get("CompressionType", "zlib")
        if compression_type == "zlib":
            data = deflate.zlib_compress(data, compression_level)
        elif compression_type == "zstandard":
            data = zstandard.compress(data, compression_level)
        else:
            raise ValueError("Unknown compression type: " + compression_type)
        h["CompressedDataSize"] = str(len(data))

    # Add "ElementDataFile" at the end
    h.pop("ElementDataFile", None)
    h["ElementDataFile"] = data_filename
    with open(filename, "w") as f:
        for key, value in h.items():
            f.write(key + " = " + value + "\n")

    if data_filename == "LOCAL":
        with open(filename, "ab") as fdata:
            fdata.write(data)
    else:
        with open(
            os.path.join(os.path.dirname(filename), data_filename), "wb"
        ) as fdata:
            fdata.write(data)

RESERVE_SIZE = 64  # reserve space for compressed data size

class Writer:
    def __init__(self, filename: Union[Path, str], header: Dict[str, Any], dtype):
        self.filename = filename
        header["ElementType"] = _DTYPE2METATYPE_TABLE[dtype.name]
        self.compress = header["CompressedData"] == "True"
        if Path(filename).suffix == ".mhd":
            if self.compress:
                data_filename = Path(filename).stem + ".zraw"
            else:
                data_filename = Path(filename).stem + ".raw"
        else:
            if Path(filename).suffix != ".mha":
                warnings.warn(
                    'Unknown file extension "{0}". Saving as a .mha file.'.format(
                        Path(filename).suffix
                    ),
                    stacklevel=2,
                )
            data_filename = "LOCAL"
        if self.compress:
            header['CompressedDataSize'] = ' ' * RESERVE_SIZE # reserve space for compressed data size
            compression_type = header.get("CompressionType", "zlib")
            if compression_type == "zstandard":
                self.compressor = zstandard.ZstdCompressor().compressobj()
            elif compression_type == "zlib":
                self.compressor = zlib.compressobj()
            else:
                raise ValueError("Unknown compression type: " + compression_type)
        else:
            self.compressor = None
        header["ElementDataFile"] = data_filename
        with open(filename, "w") as f:
            for key, value in header.items():
                value = _array2str(value)
                line = f'{key} = {value}\n'
                f.write(line)
        self.total_size = 0

    def write(self, data: np.ndarray):
        if self.compress:
            data = self.compressor.compress(data.tobytes())
        self.total_size += len(data)
        with open(self.filename, "ab") as fdata:
            fdata.write(data)

    def close(self):
        if self.compress:
            data = self.compressor.flush()
            self.total_size += len(data)
            with open(self.filename, "ab") as fdata:
                fdata.write(data)
            with open(self.filename, "r+") as f:
                # Use mmap to update CompressedDataSize
                mm = mmap(f.fileno(), 0)
                # Find the position of CompressedDataSize
                pos = mm.find(b"CompressedDataSize")
                if pos == -1:
                    raise RuntimeError("CompressedDataSize not found in header")
                line = f'CompressedDataSize = {self.total_size}'
                bytes = line.encode('ascii')

                mm[pos:pos + len(bytes)] = bytes
                mm.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

def reorient(volume: np.ndarray, header: Dict[str, Any]):
    """
    (WIP) Re-orient volume so that transform matrix become closer to [[1,0,0], [0,1,0], [0,0,1]]
    """
    assert volume.ndim == 3
    tm = header["TransformMatrix"]
    tm = np.array(tm).reshape((3, 3))[::-1]
    eye = np.eye(3)[::-1]
    for i in range(3):
        ip = np.inner(tm[i], eye[i])
        if ip < 0:
            volume = np.flip(volume, axis=i)
    return volume


class ImageIterator:

    def __init__(
        self, filename: Union[str, Path], shape: List[int], dtype: np.dtype, seek_size
    ):
        self.filename = filename
        self.shape = shape
        self.dtype = dtype
        self.seek_size = seek_size
        self.size_per_image = int(np.prod(shape[1:])) * int(dtype.itemsize)
        self.i = 0

    def __iter__(self) -> Iterator[np.ndarray]:
        return self

    def __next__(self):
        if self.i >= self.shape[0]:
            raise StopIteration()
        data = bytearray(self.size_per_image)
        with open(self.filename, "rb") as f:
            f.seek(self.seek_size + self.i * self.size_per_image)
            f.readinto(data)
        self.i = self.i + 1
        arr = np.frombuffer(data, dtype=self.dtype)
        arr = np.reshape(arr, self.shape[1:], order="C")
        try:
            arr.setflags(write=True)
        except Exception:
            pass
        return arr

class ZlibIterator:
    def __init__(self, data, window_size: int):
        self.data = data
        self.size = window_size
        self.decomp_iter = zlib.decompressobj()

    def __iter__(self) -> Iterator[np.ndarray]:
        return self

    def __next__(self):
        if not self.data:
            raise StopIteration()
        data = self.decomp_iter.decompress(self.data, max_length=self.size)
        self.data = self.decomp_iter.unconsumed_tail
        return data

class ZstandardIterator:
    def __init__(self, data, window_size: int):
        self.data = data
        self.size = window_size
        self.stream_reader = zstandard.ZstdDecompressor().stream_reader(
            data
        )

    def __iter__(self) -> Iterator[np.ndarray]:
        return self

    def __next__(self):
        data = self.stream_reader.read(self.size)
        if not data:
            raise StopIteration()
        return data

class CompressedImageIterator:

    def __init__(
        self, filename: Union[str, Path], shape: List[int], dtype: np.dtype, seek_size: int, compression_type: str
    ):
        self.filename = filename
        self.shape = shape
        self.dtype = dtype
        self.size_per_image = int(np.prod(shape[1:])) * int(dtype.itemsize)
        with open(filename, "rb") as f:
            f.seek(seek_size)
            self.compressed_bytes = f.read()
        if compression_type == 'zstandard':
            self.decomp_iter = ZstandardIterator(self.compressed_bytes, self.size_per_image)
        elif compression_type == 'zlib':
            self.decomp_iter = ZlibIterator(self.compressed_bytes, self.size_per_image)
        else:
            raise ValueError("Unknown compression type: " + compression_type)

    def __iter__(self) -> Iterator[np.ndarray]:
        return self

    def __next__(self):
        data = next(self.decomp_iter)
        data = bytearray(data)
        arr = np.frombuffer(data, dtype=self.dtype)
        arr = np.reshape(arr, self.shape[1:], order="C")
        try:
            arr.setflags(write=True)
        except Exception:
            pass
        return arr


def read_iterator(
    filename: Union[Path, str], encoding="ascii"
) -> Tuple[Iterator[np.ndarray], Dict[str, Any]]:
    """Read as an iterator that reads one image at a time from ndarray.
    e.g. if original array is [z,y,x], the iterator returns [y,x] array.
    For uncompressed data, bytes for one image is allocated per iteration.
    For compressed data, bytes for the entire compressed data is consumed in addition to the per-iteration consumption.

    :param str filename: Image filename with extension mhd or mha.
    :return: iterator and metadata
    :rtype: (iterator, dict)

    Examples:
        >>> import mhd
        >>> image_iterator, header = mhd.read('filename.mhd')
        >>> for image in image_iterator:
        >>>     do_some_stuff(image)
    """
    filename = str(filename)
    header = read_header(filename, encoding)
    data_is_compressed = "CompressedData" in header and header["CompressedData"]
    data_filename = header["ElementDataFile"]
    if header["ObjectType"] != "Image":
        raise ValueError('ObjectType not "Image" is not supported (yet.)')
    if data_filename == "LIST":
        raise ValueError('ElementDataFile "LIST" is not supported (yet.)')
    if data_filename == "LOCAL":  # mha
        data_filename = filename
        if data_is_compressed:
            data_size = header["CompressedDataSize"]
        else:
            numel = np.prod(_get_dim(header))
            data_size = int(numel) * int(
                np.dtype(_METATYPE2DTYPE_TABLE[header["ElementType"]]).itemsize
            )
        seek_size = os.path.getsize(filename) - data_size
    else:  # mhd
        if not os.path.isabs(data_filename):  # data_filename is relative
            data_filename = os.path.join(os.path.dirname(filename), data_filename)
        seek_size = 0
    dim = _get_dim(header)
    shape = list(reversed(dim))
    dtype = np.dtype(_METATYPE2DTYPE_TABLE[header["ElementType"]])
    if data_is_compressed:
        compression_type = header.get("CompressionType", "zlib")
        iterator = CompressedImageIterator(data_filename, shape, dtype, seek_size, compression_type)
    else:
        iterator = ImageIterator(data_filename, shape, dtype, seek_size)
    return iterator, header
