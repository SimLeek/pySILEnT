# coding:utf-8

"""
This code generates tensors for keeping the total brightness constant in an area x pixels around any pixel.
"""

import tensorflow as tf


def regulate_tensor(input_tensor,
                    blur_tensor,
                    regulation_value,
                    regulation_root=1.0 / 2.0,
                    strides=(1, 1, 1, 1),
                    padding='SAME'
                    ):
    """ Uses a blurred version of the input tensor to perform "adaptive thresholding", except it boosts or inhibits
    instead, since there's no threshold.

    When used on an edge_orientation_detector or stripe detector, this can brighten edges that are faint, but in dark or
     blurry parts of the mage. It will also dim edges that are bright, but near bright or high contrast parts of an
     imaage.

    :param input_tensor: the tensor or image to regulate
    :param blur_tensor: the blurred version of the tensor or image
    :param regulation_value: A constant regulation value that will be multiplied to all parts of the image.
    :param regulation_root: Power to which each pixel, voxel, etc. of the blurred input will be raised to. 1 will ensure
     all lines in the image have equal brightness. 0.5 will enhance dark lines, but still let bright lines stand out.
     0 will make this an expensive constant multiplication operation.
    :param strides: Tensorflow conv2d strides.
    :param padding: Tensorflow conv2d padding.
    :return: A regulated version of the input tensor.
    """
    blurred_input = tf.nn.conv2d(input=input_tensor, filter=blur_tensor, strides=strides, padding=padding)
    regulator_tensor = regulation_value / tf.pow(tf.minimum(blurred_input, [1]), regulation_root)
    return input_tensor * regulator_tensor
