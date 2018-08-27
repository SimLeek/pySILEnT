from slam_recognition import center_surround_tensor
import unittest as ut
import numpy.testing as npt
import time


class TestCenterSurroundTensor(ut.TestCase):
    def test_surround_center(self):
        test_1d = center_surround_tensor(1, [0, 1, 0], [1, 0, 0],
                                            [0, 0, 1], [1, 0, 0])

        npt.assert_array_equal(test_1d, [[[0., 0., 0., ],
                                          [0., 0., 0., ],
                                          [1., 0., 0., ]],

                                         [[0., 0., 0.],
                                          [2., 0., 0.],
                                          [0., 0., 0.]],

                                         [[0., 0., 0.],
                                          [0., 0., 0.],
                                          [1., 0., 0.]]])

    def test_surround_center_2d(self):
        test_2d = center_surround_tensor(2, [0, 1, 0], [1, 0, 0],
                                         [0, 0, 1], [1, 0, 0])
        npt.assert_array_almost_equal(test_2d,
                                      [[[[0., 0., 0.],
                                         [0., 0., 0.],
                                         [0.70710678, 0., 0.]],

                                        [[0., 0., 0.],
                                         [0., 0., 0.],
                                         [1., 0., 0.]],

                                        [[0., 0., 0.],
                                         [0., 0., 0.],
                                         [0.70710678, 0., 0.]]],

                                       [[[0., 0., 0.],
                                         [0., 0., 0.],
                                         [1., 0., 0.]],

                                        [[0., 0., 0.],
                                         [6.82842712, 0., 0.],
                                         [0., 0., 0.]],

                                        [[0., 0., 0.],
                                         [0., 0., 0.],
                                         [1., 0., 0.]]],

                                       [[[0., 0., 0.],
                                         [0., 0., 0.],
                                         [0.70710678, 0., 0.]],

                                        [[0., 0., 0.],
                                         [0., 0., 0.],
                                         [1., 0., 0.]],

                                        [[0., 0., 0.],
                                         [0., 0., 0.],
                                         [0.70710678, 0., 0.]]]]
                                      )

class TestRetinalFilterPerformance(ut.TestCase):
    def test_surround_center_time(self):
        for i in range(1, 11):
            t1 = time.time()
            center_surround_tensor(i, [0, 1, 0], [1, 0, 0],
                                   [0, 0, 1], [1, 0, 0])
            elapsed = time.time() - t1
            self.assertLessEqual(elapsed, 1.0,
                                 "{}-dimensional center-surround took {} seconds to generate.".format(i, elapsed))
