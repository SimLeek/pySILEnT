from slam_recognition.util.attractor import linear_attractor_function_generator as __linear_function_generator
import numpy as np
import math as m
import itertools
from slam_recognition.util.normalize import normalize_tensor_positive_negative as __normalize_center_surround
from slam_recognition.util.orientation import simplex_coordinates as __simplex_coordinates

if False:
    from typing import Union, List, Callable
    from numbers import Real


def end_tensor(end_vector,  # type: Union[np.ndarray, List[int]]
               center_in,  # type: List[int]
               center_out,  # type: List[int]
               surround_in,  # type: List[int]
               surround_out,  # type: List[int]
               attractor_function=__linear_function_generator,  # type: Callable[[], Callable[[Real], Real]]
               size=3
               ):
    """Creates a tensor where angles closer to the end vector are more positive, and angles further are more negative.

    NOTE: scaling down and detecting different lengths of lines using elongated/oriented center surrounds might be
    better.

    :param end_vector: Desired orientation to detect the most input.
    :param size: Desired width and height of tensor.
     """
    ndim = len(end_vector)
    assert ndim >= 1
    if not isinstance(end_vector, np.ndarray):
        end_vector = np.asarray(end_vector)

    attractor_function = attractor_function()

    center_surround = np.zeros(shape=[size for _ in range(ndim)] + [len(center_out), len(center_in)])
    zero_centered = np.ndarray(shape=[size for _ in range(ndim)])

    for tup in itertools.product(*[range(size) for _ in range(ndim)]):
        tup_vec = np.asarray(tup) - np.asarray([size / 2 for _ in range(ndim)])
        angle_dist = (m.acos(
            np.dot(tup_vec, end_vector) / (np.linalg.norm(tup_vec) * np.linalg.norm(end_vector))) - m.pi / 2.0) / m.pi
        zero_centered[tup] = attractor_function(angle_dist * m.pi)

    __normalize_center_surround(zero_centered)

    for tup in itertools.product(*[range(size) for _ in range(ndim)]):
        center_surround[tup] = [[surround_out[o] * surround_in[i] * abs(zero_centered[tup]) if zero_centered[tup] < 0
                                 else center_surround[tuple(tup + (o, i))]
                                 for o in range(len(surround_out))] for i in range(len(surround_in))]
        center_surround[tup] = [[center_out[o] * center_in[i] * abs(zero_centered[tup]) if zero_centered[tup] >= 0
                                 else center_surround[tuple(tup + (i, o))]
                                 for o in range(len(center_out))] for i in range(len(center_in))]

    return center_surround


def simplex_end_tensors(dimension,  # type: int
                        centers_in,  # type: List[List[int]]
                        centers_out,  # type: List[List[int]]
                        surrounds_in,  # type: List[List[int]]
                        surrounds_out,  # type: List[List[int]]
                        attractor_function=__linear_function_generator,  # type: Callable[[...], Callable[[Real], Real]]
                        flip=True
                        ):
    """ Generates all tensors needed to find all line-ends in the specified number of dimensions.

    :param flip: Determines whether we flip our simplex vectors or not.
    :return: all tensors for detecting line ends.
    """
    simplex = __simplex_coordinates(dimension)

    simplex *= 3

    if flip is not None:
        # Flip works better:
        # simplex = np.negative(simplex)
        simplex = np.flip(simplex, flip)

    return [end_tensor(simplex_vector, center_in, center_out, surround_in, surround_out, attractor_function)
            for simplex_vector, center_in, center_out, surround_in, surround_out
            in zip(simplex, centers_in, centers_out, surrounds_in, surrounds_out)]


def rgb_2d_end_tensors(north_input_channel=(1, 0, 0),
                       southwest_input_channel=(0, 1, 0),
                       southeast_input_channel=(0, 0, 1)):
    """ Generates all tensors needed to find all line-ends in 2 dimensions."""
    x = 0.5 / 2
    xx = -0.25 / 2
    y = 1.0 / 2
    yy = 1.0 / 2

    end_tensors = sum(simplex_end_tensors(2, [north_input_channel, southwest_input_channel, southeast_input_channel],
                                          [[x, -xx, -xx], [-xx, x, -xx], [-xx, -xx, x]],
                                          [north_input_channel, southwest_input_channel, southeast_input_channel],
                                          [[y, -yy, -yy], [-yy, y, -yy], [-yy, -yy, y]]))

    return end_tensors
