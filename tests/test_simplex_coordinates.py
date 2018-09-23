import unittest as ut

import numpy.testing as nptest

from slam_recognition.util.orientation import simplex_coordinates


class TestSimplexCoordinates(ut.TestCase):
    def test_simplex_2d(self):
        s2 = simplex_coordinates(2)
        nptest.assert_array_almost_equal(s2,
                                         [[1., 0.],
                                          [-0.5, 0.8660254],
                                          [-0.5, - 0.8660254]]
                                         )

        s3 = simplex_coordinates(3)
        nptest.assert_array_almost_equal(s3,
                                         [[1., 0., 0.],
                                          [-0.33333333, 0.94280904, 0.],
                                          [-0.33333333, - 0.47140452, 0.81649658],
                                          [-0.33333333, - 0.47140452, - 0.81649658]])
