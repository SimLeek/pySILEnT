# coding:utf-8

"""
Greatly enhances monochromatic colors in images. Mimics color opponent center surround cells in human visual cortex V1.
"""

from . import center_surround_tensor
from slam_recognition.util.normalize import normalize_tensor_positive_negative

if False:
    import numpy as np


def rgby(n  # type: int
         ):  # type: (...)->np.ndarray
    """Returns a tensor that can convolve a color image for better edge_orientation_detector detection.
    Based off visual cortex V1.

    :param n: number of dimensions
    :return: tensor used for convolution
    """
    d = 1.
    out = \
        center_surround_tensor(n, center_in=[0, 0, d], center_out=[0, 0, d, 0],  # red green
                               surround_in=[0, d, 0], surround_out=[0, 0, -d, 0]) + \
        center_surround_tensor(n, center_in=[d, 0, 0], center_out=[d, 0, 0, 0],  # blue yellow
                               surround_in=[0, d / 2, d / 2], surround_out=[-d, 0, 0, 0]) + \
        center_surround_tensor(n, center_in=[0, d, 0], center_out=[0, d, 0, 0],  # green red
                               surround_in=[0, 0, d], surround_out=[0, -d, 0, 0]) + \
        center_surround_tensor(n, center_in=[0, d / 2, d / 2], center_out=[0, 0, 0, d],
                               # yellow blue
                               surround_in=[d, 0, 0], surround_out=[0, 0, 0, -d])
    return out


def rgby_3(n):
    """Returns a tensor that can convolve a color image for better edge_orientation_detector detection.
    Based off visual cortex V1.

    This version results in a three color image, so humans can see and debug the output.

    :param n: number of dimensions
    :return: tensor used for convolution
    """
    d = 1. / 3
    out = \
        center_surround_tensor(n, center_in=[0, 0, d], center_out=[0, 0, d],  # red green
                               surround_in=[0, d, 0], surround_out=[0, 0, -d]) + \
        center_surround_tensor(n, center_in=[d, 0, 0], center_out=[d, 0, 0],  # blue yellow
                               surround_in=[0, d / 2, d / 2], surround_out=[-d, 0, 0]) + \
        center_surround_tensor(n, center_in=[0, d, 0], center_out=[0, d, 0],  # green red
                               surround_in=[0, 0, d], surround_out=[0, -d, 0]) + \
        center_surround_tensor(n, center_in=[0, d / 2, d / 2], center_out=[0, d / 2, d / 2],
                               # yellow blue
                               surround_in=[d, 0, 0], surround_out=[0, -d / 2, -d / 2])
    return normalize_tensor_positive_negative(out, 4.0, 2.0)
