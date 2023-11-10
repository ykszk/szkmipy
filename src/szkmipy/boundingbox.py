'''
Compute bounding boxes and crop arrays

Examples:
    >>> import szkmipy.boundingbox as bb
    >>> arr = np.array([[1,2,3], [4,5,6], [0, 0, 0]])
    >>> bbox = bb.bbox(arr)
    >>> bbox
    (array([0, 0]), array([2, 3]))
    >>> cropped_arr = bb.crop(arr, bbox, margin=0)
    >>> cropped_arr.shape
    (2, 3)
    >>> np.array_equal(cropped_arr, bb.trim(arr, margin=0))
    True
    >>> processed_cropped_arr = cropped_arr ** 2
    >>> processed = bb.uncrop(processed_cropped_arr, arr.shape, bbox, margin=0)
    >>> processed.shape == arr.shape
    True
    >>> bb.bbox_mask_array(arr)
    array([[1, 1, 1],
           [1, 1, 1],
           [0, 0, 0]], dtype=uint8)

'''
from typing import Tuple, Union

import numpy as np
import numpy.typing as npt

BBox = Tuple[np.ndarray, np.ndarray]


def bbox(arr: np.ndarray) -> BBox:
    """
    Compute bounding box for given ndarray.

    Args:
        arr (ndarray): Input array.
    Returns:
        (np.array, np.array): Bounding box of input array. (bbox_min, bbox_max)
    """
    if np.count_nonzero(arr) == 0:
        raise ValueError('Input array is empty.')
    dim = arr.ndim
    bb = np.array([
        np.nonzero(np.any(arr, axis=tuple([i for i in range(dim)
                                           if i != d])))[0][[0, -1]]
        for d in range(dim)
    ])
    bb[:, 1] += 1
    return bb[:, 0], bb[:, 1]


def crop(arr: np.ndarray, bbox: BBox, margin: Union[int, npt.ArrayLike]=0) -> np.ndarray:
    """
    Crop array using given bounding box.

    Args:
        arr (ndarray): Input array.
        bbox (np.array, np.array): Input bounding box.
        margin (int): The size of margin.
    Returns:
        np.ndarray: Cropped array.
    """
    bmin = bbox[0]
    bmax = bbox[1]
    if hasattr(margin, '__len__'):
        v_margin = np.array(margin)
    else:
        v_margin = np.repeat(margin, len(bmin))
    bmin = np.maximum(0, bmin - v_margin)
    bmax = np.minimum(np.array(arr.shape), bmax + v_margin)
    a = arr[tuple([slice(bmin[i], bmax[i]) for i in range(len(bmin))])]
    return a


def trim(arr: np.ndarray, margin: Union[int, npt.ArrayLike] = 0) -> np.ndarray:
    """
    Trim array.

    This function is equivalent to ``crop(array, bbox(array), margin)``

    Args:
        arr (ndarray): Input array.
        margin (int): The size of margin.
    Returns:
        np.ndarray: Trimmed array.
    """
    bb = bbox(arr)
    return crop(arr, bb, margin)


def uncrop(arr: np.ndarray,
           original_shape: npt.ArrayLike,
           bbox: BBox,
           margin: int = 0,
           constant_values: int = 0) -> np.ndarray:
    '''
    Revert cropping

    Args:
        ararrray (ndarray): Input cropped array.
        original_shape (array__like): Original shape before cropping.
        bbox (np.array, np.array): Bounding box used for cropping.
        margin (int): Margin used for cropping.
        constant_values (int or array_like): Passed to np.pad
    Returns:
        np.ndarray: Uncropped array.
    '''
    start = np.maximum(bbox[0] - margin, 0)
    end = np.maximum(np.array(original_shape) - bbox[1] - margin, 0)
    pad_width = np.array((start, end)).T
    return np.pad(arr, pad_width, 'constant', constant_values=constant_values)


def bbox_mask_array(array: np.ndarray, margin: int = 0) -> np.ndarray:
    '''
    Create bounding box mask array
    '''
    bbarray = np.zeros(array.shape, dtype=np.uint8)
    bb = bbox(array)
    bbmin = np.clip(bb[0] - margin, 0, None)
    bbmax = np.clip(bb[1] + margin, None, bbarray.shape)
    bbarray[tuple([slice(*bb) for bb in zip(bbmin, bbmax)])] = 1
    return bbarray
