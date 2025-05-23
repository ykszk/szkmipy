import unittest
import szkmipy.boundingbox as bb
import numpy as np
from numpy import testing
from logging import getLogger
logger = getLogger(__name__)

test_torch = False
try:
    import torch
    test_torch = True
except ImportError:
    pass

class TestBoundingbox(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestBoundingbox, self).__init__(*args, **kwargs)
        if test_torch:
            logger.info('Testing torch implementation')
            bb.enable_torch()

    @classmethod
    def get_2d_arr(cls):
        arr = np.zeros([5, 5])
        arr[1, 1] = 1
        arr[2, 3] = 1
        return arr

    def test_bbox(self):
        arr = self.get_2d_arr()
        bmin, bmax = bb.bbox(arr)
        testing.assert_array_equal(np.array([1, 1]), bmin)
        testing.assert_array_equal(np.array([3,4]), bmax)
        if test_torch:
            arr_torch = torch.tensor(arr, dtype=torch.bool)
            pt_bmin, pt_bmax = bb.bbox(arr_torch)
            testing.assert_array_equal(bmin, pt_bmin)
            testing.assert_array_equal(bmax, pt_bmax)

    def test_trim(self):
        arr = self.get_2d_arr()

        for margin, expected in zip([0, 1, 2], [arr[1:3, 1:4], arr[0:4, 0:5], arr]):
            trimmed = bb.trim(arr, margin=margin)
            testing.assert_array_equal(expected, trimmed)
            if test_torch:
                trimmed_torch = bb.trim(torch.tensor(arr, dtype=torch.bool), margin=margin)
                testing.assert_array_equal(trimmed, trimmed_torch.numpy())


    def test_uncrop(self):
        arr = self.get_2d_arr()
        bbox = bb.bbox(arr)
        for margin in range(4):
            cropped = bb.crop(arr, bbox, margin=margin)
            testing.assert_array_equal(
                arr, bb.uncrop(cropped, arr.shape, bbox, margin=margin))
            if test_torch:
                cropped_torch = bb.crop(torch.tensor(arr), bbox, margin=margin)
                testing.assert_array_equal(
                    cropped, cropped_torch.numpy())
                testing.assert_array_equal(
                    arr, bb.uncrop(cropped_torch, arr.shape, bbox, margin=margin))


    def test_add_margin_to_bbox(self):
        arr = self.get_2d_arr()
        bbox = bb.bbox(arr)
        
        # Test with integer margin
        new_bbox = bb.add_margin_to_bbox(arr, bbox, 1)
        expected_min = np.array([0, 0])
        expected_max = np.array([4, 5])
        testing.assert_array_equal(expected_min, new_bbox[0])
        testing.assert_array_equal(expected_max, new_bbox[1])
        
        # Test with array-like margin
        new_bbox = bb.add_margin_to_bbox(arr, bbox, [2, 1])
        expected_min = np.array([0, 0])
        expected_max = np.array([5, 5])
        testing.assert_array_equal(expected_min, new_bbox[0])
        testing.assert_array_equal(expected_max, new_bbox[1])