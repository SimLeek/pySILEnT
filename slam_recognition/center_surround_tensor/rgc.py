# coding:utf-8

"""
Here we mimic the midget retinal ganglian cells of the human eye.
"""

from slam_recognition.util.normalize import normalize_tensor_positive_negative
from . import center_surround_tensor

if False:
    import numpy as np


def midget_rgc(n  # type: int
               ):  # type: (...)->np.ndarray
    """Returns a tensor that can convolve a color image for better edge detection.

    Based off of human retinal ganglian cells.

    :param n: number of dimensions
    :return: tensor used for convolution
    """
    d = 1. / 2
    out = \
        center_surround_tensor(n, center_in=[d, 0, 0], center_out=[d, 0, 0],
                               surround_in=[d / 2, 0, 0], surround_out=[-d, 0, 0]) + \
        center_surround_tensor(n, center_in=[0, d, 0], center_out=[0, d, 0],
                               surround_in=[0, d / 2, 0], surround_out=[0, -d, 0]) + \
        center_surround_tensor(n, center_in=[0, 0, d], center_out=[0, 0, d],
                               surround_in=[0, 0, d / 2], surround_out=[0, 0, -d])

    return normalize_tensor_positive_negative(out, 4.0, 2.0)
