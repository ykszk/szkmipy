import unittest
import tempfile
from pathlib import Path

from szkmipy import mhd
import numpy as np
from numpy import testing


class TestMhd(unittest.TestCase):

    def setUp(self, *args, **kwargs):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        self.data_list = []
        dtypes = [
            np.uint8, np.int16, np.uint16, np.int32, np.float32, np.float64
        ]
        size = 8
        for dtype in dtypes:  # 2D
            self.data_list.append(
                (np.random.rand(size, size) * 100).astype(dtype))
        for dtype in dtypes:  # 3D
            self.data_list.append(
                (np.random.rand(size, size, size) * 100).astype(dtype))

    def tearDown(self):
        self.temp_dir.cleanup()
        del self.temp_dir

    def test_simple_io(self):
        for filename in ['tmp.mhd', 'tmp.mha']:
            for data in self.data_list:
                for compression in [True, False]:
                    filepath = self.temp_path / filename
                    mhd.write(filepath, data, {'CompressedData': compression})
                    data_read, header = mhd.read(filepath)
                    testing.assert_array_equal(data, data_read)
                    self.assertEqual(compression, header['CompressedData'])

    def test_multi_channel(self):
        filepath = self.temp_path / 'tmp.mha'
        size = (16, 8, 32, 3)
        data = np.random.rand(*size)
        mhd.write(filepath, data, {'ElementNumberOfChannels': size[-1]})
        data_read, header = mhd.read(filepath)
        testing.assert_array_equal(data, data_read)
        self.assertEqual(header['ElementNumberOfChannels'],
                         data_read.shape[-1])
        self.assertEqual(size[-1], data_read.shape[-1])

    def test_read_header(self):
        data = self.data_list[-1]
        for filename, data_filename in [['tmp.mhd', 'tmp.raw'],
                                        ['tmp.mha', 'LOCAL']]:
            filepath = self.temp_path / filename
            mhd.write(filepath, data)
            header = mhd.read_header(filepath)
            self.assertEqual(header['NDims'], data.ndim)
            self.assertEqual(header['DimSize'], list(data.shape[::-1]))
            self.assertEqual(header['ElementType'],
                             mhd._DTYPE2METATYPE_TABLE[data.dtype.name])
            self.assertEqual(header['ElementDataFile'], data_filename)

    def test_mutability(self):
        filepath = self.temp_path / 'tmp.mha'
        mhd.write(filepath, self.data_list[0], {'CompressedData': False})
        data_read, _ = mhd.read(filepath)
        data_read[0] = 0

    def test_meta_io(self):
        for filename in ['tmp.mhd', 'tmp.mha']:
            filepath = self.temp_path / filename
            data = self.data_list[0]
            mhd.write(filepath, data)
            header1 = mhd.read_header(filepath)

            # add new meta data
            ed = header1.pop('ElementDataFile')
            header1['key1'] = 'value1'
            header1['key2'] = 'value2'
            header1['ElementDataFile'] = ed

            mhd.write(filepath, self.data_list[0], header1)
            data_read, header2 = mhd.read(filepath)
            testing.assert_array_equal(data, data_read)
            self.assertEqual(header1, header2)

    def test_memmap(self):
        for filename in ['tmp.mhd', 'tmp.mha']:
            filepath = self.temp_path / filename
            for data in self.data_list:
                mhd.write(filepath, data, {'CompressedData': False})
                data_read = mhd.read_memmap(filepath)[0]
                testing.assert_array_equal(data, data_read)
                del data_read

    def test_iterator(self):
        for filename in ['tmp.mhd', 'tmp.mha']:
            filepath = self.temp_path / filename
            for data in self.data_list:
                for compression in [False, True]:
                    mhd.write(filepath, data, {'CompressedData': compression})
                    iter = mhd.read_iterator(filepath)[0]
                    for data_orig, data_read in zip(data, iter):
                        testing.assert_array_equal(data_orig, data_read)


if __name__ == "__main__":
    unittest.main()
