from slam_recognition import center_surround_tensor
from slam_recognition.util.normalize import normalize_tensor_positive_negative
import unittest as ut
import numpy.testing as nptest
import numpy as np


class TestNormalizeCenterSurround(ut.TestCase):
    def test_normalize_basic_center_surround(self):
        test_1d = np.squeeze(center_surround_tensor(1, [1], [1], [1], [-1]))
        nptest.assert_array_almost_equal(test_1d, [-1, 2, -1])

        test_norm = normalize_tensor_positive_negative(test_1d)
        nptest.assert_array_almost_equal(test_norm, [-.5, 1, -.5])

        test_2d = np.squeeze(center_surround_tensor(2, [1], [1], [1], [-1]))
        nptest.assert_array_almost_equal(test_2d, [[-0.70710678, -1., -0.70710678],
                                                   [-1., 6.82842712, -1.],
                                                   [-0.70710678, -1., -0.70710678]])
        test_norm_2d = normalize_tensor_positive_negative(test_2d)
        nptest.assert_array_almost_equal(test_norm_2d, [[-0.10355339, -0.14644661, -0.10355339],
                                                        [-0.14644661, 1., -0.14644661],
                                                        [-0.10355339, -0.14644661, -0.10355339]])
        nptest.assert_array_almost_equal(test_2d,      [[-0.10355339, -0.14644661, -0.10355339],
                                                        [-0.14644661, 1., -0.14644661],
                                                        [-0.10355339, -0.14644661, -0.10355339]])
