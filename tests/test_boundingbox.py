import unittest
import szkmipy.boundingbox as bb
import numpy as np
from numpy import testing


class TestBoundingbox(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestBoundingbox, self).__init__(*args, **kwargs)

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
        testing.assert_array_equal(np.array([2, 3]), bmax)

    def test_trim(self):
        arr = self.get_2d_arr()
        trimmed = bb.trim(arr)
        testing.assert_array_equal(arr[1:3, 1:4], trimmed)
        trimmed = bb.trim(arr, margin=1)
        testing.assert_array_equal(arr[0:4, 0:5], trimmed)
        trimmed = bb.trim(arr, margin=2)
        testing.assert_array_equal(arr, trimmed)

    def test_uncrop(self):
        arr = self.get_2d_arr()
        bbox = bb.bbox(arr)
        for margin in range(4):
            cropped = bb.crop(arr, bbox, margin=margin)
            testing.assert_array_equal(
                arr, bb.uncrop(cropped, arr.shape, bbox, margin=margin))
