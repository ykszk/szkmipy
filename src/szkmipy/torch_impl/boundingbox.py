'''
PyTorch implementation of boundingbox module
'''
from typing import Union
import torch
import numpy as np
import numpy.typing as npt
from .. import boundingbox as bb


def bbox(arr: torch.BoolTensor) -> bb.BBox:
    if torch.count_nonzero(arr) == 0:
        raise ValueError('Input array is empty.')
    dim = arr.ndim
    bb = np.array(
        [
            torch.nonzero(torch.any(arr, dim=tuple([i for i in range(dim) if i != d]), keepdim=False))[:, 0][[0, -1]]
            for d in range(dim)
        ]
    )
    bb[:, 1] += 1
    return bb[:, 0], bb[:, 1]

def crop(arr: torch.Tensor, bbox: bb.BBox, margin: Union[int, npt.ArrayLike] = 0) -> torch.Tensor:
    bmin = torch.tensor(bbox[0])
    bmax = torch.tensor(bbox[1])
    if hasattr(margin, '__len__'):
        v_margin = torch.tensor(margin)
    else:
        v_margin = torch.repeat_interleave(torch.tensor(margin), len(bmin))
    bmin = torch.maximum(torch.tensor(0), bmin - v_margin)
    bmax = torch.minimum(torch.tensor(arr.shape), bmax + v_margin)
    a = arr[tuple([slice(bmin[i], bmax[i]) for i in range(len(bmin))])]
    return a


def trim(arr: torch.Tensor, margin: Union[int, npt.ArrayLike] = 0) -> torch.Tensor:
    return crop(arr, bbox(arr), margin)

def uncrop(cropped: torch.Tensor, original_shape: npt.ArrayLike, bbox: bb.BBox, margin: Union[int, npt.ArrayLike] = 0, constant_values: int = 0) -> torch.Tensor:
    start = np.maximum(bbox[0] - margin, 0)
    end = np.maximum(np.array(original_shape) - bbox[1] - margin, 0)
    # pad argument is different from numpy
    # https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
    pad_width = np.array((start[::-1], end[::-1])).T
    # to flat tuple
    pad_width = tuple(pad_width.flatten
                      ())
    return torch.nn.functional.pad(cropped, pad_width, mode='constant', value=constant_values)