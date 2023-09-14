"""
Read/Write Meta Image
"""

from typing import Dict, Tuple, Union, Any
import re
from pathlib import Path
import zlib
import os
import copy
import warnings
from collections import OrderedDict

import numpy as np

_METATYPE2DTYPE_TABLE = {
    'MET_CHAR': 'i1',
    'MET_UCHAR': 'u1',
    'MET_SHORT': 'i2',
    'MET_USHORT': 'u2',
    'MET_INT': 'i4',
    'MET_UINT': 'u4',
    'MET_LONG': 'i8',
    'MET_ULONG': 'u8',
    'MET_FLOAT': 'f4',
    'MET_DOUBLE': 'f8'
}
_DTYPE2METATYPE_TABLE = {
    'int8': 'MET_CHAR',
    'uint8': 'MET_UCHAR',
    'int16': 'MET_SHORT',
    'uint16': 'MET_USHORT',
    'int32': 'MET_INT',
    'uint32': 'MET_UINT',
    'int64': 'MET_LONG',
    'uint64': 'MET_ULONG',
    'float32': 'MET_FLOAT',
    'float64': 'MET_DOUBLE'
}


def _str2bool(s: str):
    if s == 'True':
        return True
    elif s == 'False':
        return False
    raise ValueError('Non boolean string')


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
    if hasattr(array, '__iter__'):
        return ' '.join([str(e) for e in array])
    else:
        return str(array)


def read_header(filename: Union[Path, str],
                encoding: str = 'ascii') -> Dict[str, Any]:
    """Read meta image header.

    :param str filename: Image filename with extension mhd or mha.
    :return: meta data dictionary.
    :rtype: dict
    """
    filename = str(filename)
    header = OrderedDict()
    with open(filename, 'rb') as f:
        meta_regex = re.compile('(.+) = (.*)')
        for line in f:
            line = line.decode(encoding)
            if line == '\n':
                continue  # skip empty line
            match = meta_regex.match(line)
            if match:
                header[match.group(1)] = match.group(2).rstrip()
                if match.group(1) == 'ElementDataFile':
                    break
            else:
                raise RuntimeError('Bad meta header line : ' + line)
    header = OrderedDict([
        (key, _str2array(value)) for (key, value) in header.items()
    ])  # convert string into array if possible
    return header


def _get_dim(header: Dict[str, Any]):
    dim = header['DimSize']
    if ('ElementNumberOfChannels' in header):
        dim = [header['ElementNumberOfChannels']] + dim
    if not hasattr(dim, '__len__'):
        dim = [dim]
    return dim


def read_memmap(filename: Union[Path, str],
                encoding='ascii') -> Tuple[np.memmap, Dict[str, Any]]:
    """Read Meta Image as a memory-map.

    :param str filename: Image filename with extension mhd or mha.
    :return: ND image and meta data.
    :rtype: (numpy.memmap, dict)
    :raises: RuntimeError if image data is compressed
    """
    filename = str(filename)
    header = read_header(filename, encoding)
    data_is_compressed = 'CompressedData' in header and header['CompressedData']
    if data_is_compressed:
        raise RuntimeError('Memory-map cannot be created for compressed data.')
    dtype = np.dtype(_METATYPE2DTYPE_TABLE[header['ElementType']])
    data_filename = header['ElementDataFile']
    if data_filename == 'LOCAL':  #mha
        numel = np.prod(_get_dim(header))
        data_size = numel * dtype.itemsize
        offset = int(os.path.getsize(filename) - data_size)
        data_filename = filename
    else:
        offset = 0
        if not os.path.isabs(data_filename):  # data_filename is relative
            data_filename = os.path.join(os.path.dirname(filename),
                                         data_filename)
    dim = _get_dim(header)
    return np.memmap(data_filename,
                     dtype=dtype,
                     mode='r',
                     shape=tuple(dim[::-1]),
                     offset=offset), header


def read(filename: Union[Path, str],
         encoding='ascii') -> Tuple[np.ndarray, Dict[str, Any]]:
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
    data_is_compressed = 'CompressedData' in header and header['CompressedData']
    data_filename = header['ElementDataFile']
    if header['ObjectType'] != 'Image':
        raise ValueError('ObjectType not "Image" is not supported (yet.)')
    if data_filename == 'LIST':
        raise ValueError('ElementDataFile "LIST" is not supported (yet.)')
    if data_filename == 'LOCAL':  # mha
        data_filename = filename
        if data_is_compressed:
            data_size = header['CompressedDataSize']
        else:
            numel = np.prod(_get_dim(header))
            data_size = int(numel) * int(
                np.dtype(
                    _METATYPE2DTYPE_TABLE[header['ElementType']]).itemsize)
        seek_size = os.path.getsize(filename) - data_size
    else:  # mhd
        if not os.path.isabs(data_filename):  # data_filename is relative
            data_filename = os.path.join(os.path.dirname(filename),
                                         data_filename)
        data_size = os.path.getsize(data_filename)
        seek_size = 0
    if data_is_compressed:
        with open(data_filename, 'rb', buffering=0) as f:
            f.seek(seek_size)
            data = f.read()
        try:
            import pylibdeflate
            numel = int(np.prod(np.array(_get_dim(header))))
            decompressed_size = numel * np.dtype(
                _METATYPE2DTYPE_TABLE[header['ElementType']]).itemsize
            data = pylibdeflate.zlib_decompress(data, decompressed_size)
        except ImportError:
            data = zlib.decompress(data)
            data = bytearray(
                data)  # TODO: This is copying underlying memory in bytes.
    else:
        try:
            import pylibdeflate
            data = pylibdeflate.read_as_bytearray(data_filename, data_size,
                                                  seek_size)
        except ImportError:
            # Not using np.fromfile in order to set writeflag=1 later
            with open(data_filename, 'rb', buffering=0) as f:
                # Using f.readinto(bytearray) to read binary file into a mutable buffer
                # This is slightly slower than simply doing `data = f.read()`, in which data is bytes and thus immutable
                data = bytearray(data_size)
                f.seek(seek_size)
                f.readinto(data)

    data = np.frombuffer(data,
                         dtype=np.dtype(
                             _METATYPE2DTYPE_TABLE[header['ElementType']]))
    dim = _get_dim(header)
    image = np.reshape(data, list(reversed(dim)), order='C')
    try:
        image.setflags(write=True)
    except Exception:
        pass
    return image, header


_no_compression_types = set(['float32', 'float64'
                             ])  # it takes longer time to compress these types


def _is_compression_preferable(np_dtype):
    return not (np_dtype in _no_compression_types)


def write(filename: Union[Path, str],
          image: np.ndarray,
          header: Dict[str, Any] = {},
          compression_level=6):
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
    if image.dtype == np.bool8:
        image = image.astype(np.uint8)
    header = copy.deepcopy(
        header)  # copy given header because this function mutate it
    # Construct header
    h = OrderedDict()
    h['ObjectType'] = 'Image'
    # Set image dependent meta data
    h['NDims'] = image.ndim
    h['ElementType'] = _DTYPE2METATYPE_TABLE[image.dtype.name]
    h['CompressedData'] = header[
        'CompressedData'] if 'CompressedData' in header.keys(
        ) else _is_compression_preferable(image.dtype.name)
    # Remove redundant keys from given header
    for key in h.keys():
        header.pop(key, None)
    # Merge default and given headers
    h.update(header)
    if ('ElementNumberOfChannels') in h:
        h['ElementNumberOfChannels'] = image.shape[-1]
        h['DimSize'] = reversed(image.shape[:-1])
        h['NDims'] -= 1
    else:
        h['DimSize'] = reversed(image.shape)

    h = OrderedDict([(key, _array2str(value)) for (key, value) in h.items()
                     ])  # convert array into string if possible
    filename_base, file_extension = os.path.splitext(
        os.path.basename(filename))
    compress_data = h[
        'CompressedData'] == 'True'  # boolean variable for convenience
    if (file_extension == '.mhd'):
        if (compress_data):
            data_filename = filename_base + '.zraw'
        else:
            data_filename = filename_base + '.raw'
    else:
        if (file_extension != '.mha'):
            warnings.warn(
                'Unknown file extension "{0}". Saving as a .mha file.'.format(
                    file_extension),
                stacklevel=2)
        data_filename = 'LOCAL'
    data = np.ascontiguousarray(image).data
    if compress_data:
        try:
            import pylibdeflate
            data = pylibdeflate.zlib_compress(data, compression_level)
        except ImportError:
            data = zlib.compress(data, compression_level)
        h['CompressedDataSize'] = str(len(data))

    # Add "ElementDataFile" at the end
    h.pop('ElementDataFile', None)
    h['ElementDataFile'] = data_filename
    with open(filename, 'w') as f:
        for key, value in h.items():
            f.write(key + ' = ' + value + '\n')

    if data_filename == 'LOCAL':
        with open(filename, 'ab') as fdata:
            fdata.write(data)
    else:
        with open(os.path.join(os.path.dirname(filename), data_filename),
                  'wb') as fdata:
            fdata.write(data)


def reorient(volume: np.ndarray, header: Dict[str, Any]):
    '''
    (WIP) Re-orient volume so that transform matrix become closer to [[1,0,0], [0,1,0], [0,0,1]]
    '''
    assert volume.ndim == 3
    tm = header['TransformMatrix']
    tm = np.array(tm).reshape((3, 3))[::-1]
    eye = np.eye(3)[::-1]
    for i in range(3):
        ip = np.inner(tm[i], eye[i])
        if ip < 0:
            volume = np.flip(volume, axis=i)
    return volume
