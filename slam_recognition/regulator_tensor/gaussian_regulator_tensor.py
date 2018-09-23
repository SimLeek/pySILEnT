# coding:utf-8

"""
This code generates tensors for keeping the total brightness constant in an area x pixels around any pixel.
"""

import numpy as np
import itertools
import math as m
import tensorflow as tf

from slam_recognition.util.attractor import euclidian_attractor_function_generator as __euclid_function_generator

if False:
    from typing import Union, List, Callable
    from numbers import Real


def blur_tensor(n,  # type: int
                lengths=3,  # type: Union[List[int],int]
                channels_in=3,  # type: int
                channels_out=3,  # type: int
                attractor_function=__euclid_function_generator  # type: Callable[[Real], Callable[[Real], Real]]
                ):
    """Generates an n-dimensional tensor that can be convolved on any tensor to blur it.

    It should be used like:
        a = input_image
        usage_per_area = a.convolve_using(blur_tensor)
        regulator = usage_per_area/3
        regged = regulator.dot(a)

    Fun note: I made this by accident.

     :param n: number of dimensions
     :param regulation_value: value that all values in our area will add up to
     :param width: width of our oval area
     :param height: height of our oval area
     :param channels_in: number of color channels to take in.
     :param channels_out: number of color channels to output to.
     :param attractor_function:

     """
    assert n >= 1

    attractor_function = attractor_function(n, max_negative=0)

    if isinstance(lengths, int):
        gauss_dimensional_shape = [lengths for _ in range(n)]
    else:
        gauss_dimensional_shape = [lengths[i] for i in range(n)]

    gauss = np.ndarray(shape=gauss_dimensional_shape + [channels_in, channels_out])

    for tup in itertools.product(*[range(gauss_dimensional_shape[i]) for i in range(n)]):
        vector_from_center = [(tup[t] - int(gauss_dimensional_shape[t] / 2)) for t in range(len(tup))]
        euclidian_distance = m.sqrt(sum([d ** 2 for d in vector_from_center]))

        for i in itertools.product(range(channels_in), range(channels_out)):
            gauss[tup + i] = attractor_function(euclidian_distance)

    return gauss


def regulate_tensor(input_tensor,
                    blur_tensor,
                    regulation_value,
                    regulation_root=1.0 / 2.0,
                    strides=(1, 1, 1, 1),
                    padding='SAME'
                    ):
    """ Uses a blurred version of the input tensor to perform "adaptive thresholding", except it boosts or inhibits
    instead, since there's no threshold.

    When used on an edge or stripe detector, this can brighten edges that are faint, but in dark or blurry parts of the
    image. It will also dim edges that are bright, but near bright or high contrast parts of an imaage.

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
