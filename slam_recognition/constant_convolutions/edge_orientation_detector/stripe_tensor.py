# coding:utf-8

"""
This code generates tensors for detecting thin edges and boundaries, and their orientations.
"""

import itertools
import math as m

import numpy as np

from slam_recognition.util.attractor import euclidian_attractor_function_generator as __euclid_function_generator
from slam_recognition.util.normalize import normalize_tensor_positive_negative as __normalize_center_surround
from slam_recognition.util.orientation import above_axis_simplex_coordinates as __axis_coordinates

if False:
    from typing import List, Callable
    from numbers import Real


def stripe_tensor(normal_vector,  # type: List[int]
                  center_in,  # type: List[int]
                  center_out,  # type: List[int]
                  surround_in,  # type: List[int]
                  surround_out,  # type: List[int]
                  attractor_function=__euclid_function_generator,  # type: Callable[[Real], Callable[[Real], Real]]
                  ):
    """Generates a multi-channel stripe tensor. These will isolate the n-1 boundaries, or facets, in n-dimensional
     space. In 3d, they will find faces; in 2d, lines; and in 1d, points. Unlike edge_orientation_detector tensors, they will only find these
     features if they have the correct thickness.

    Note: stripe tensors with 11 or more dimensions may take a while to generate. Make sure you cache those.

    :param attractor_function: function that determines the weights of each point in the tensor based on its distance
     from the central facet.
    :param normal_vector: unit vector pointing outwards from the facet/face/edge_orientation_detector.
    :param center_in: colors added together on points on the edge_orientation_detector.
    :param center_out: colors outputted on points on the edge_orientation_detector.
    :param surround_in: colors subtracted together on points off the edge_orientation_detector
    :param surround_out: colors outputted on points off the edge_orientation_detector.

    """
    assert len(normal_vector) >= 1
    ndim = len(normal_vector)
    attractor_function = attractor_function(ndim)
    if isinstance(normal_vector, list):
        normal_vector = np.asarray(normal_vector)

    center_surround = np.zeros(shape=[3 for _ in range(ndim)] + [len(center_out), len(center_in)])

    zero_centered = np.ndarray(shape=[3 for _ in range(ndim)])

    for tup in itertools.product(*[range(3) for _ in range(ndim)]):
        scalar_projection = sum([(t - 1) * n for t, n in zip(tup, normal_vector)])
        projection = normal_vector * scalar_projection
        euclidian_dist = m.sqrt(sum([p ** 2 for p in projection]))
        zero_centered[tup] = attractor_function(euclidian_dist)

    __normalize_center_surround(zero_centered)

    for tup in itertools.product(*[range(3) for _ in range(ndim)]):
        center_surround[tup] = [[surround_out[o] * surround_in[i] * abs(zero_centered[tup]) if zero_centered[tup] < 0
                                 else center_surround[tuple(tup + (o, i))]
                                 for o in range(len(surround_out))] for i in range(len(surround_in))]

        center_surround[tup] = [[center_out[o] * center_in[i] * abs(zero_centered[tup]) if zero_centered[tup] > 0
                                 else center_surround[tuple(tup + (i, o))]
                                 for o in range(len(center_out))] for i in range(len(center_in))]

    return center_surround


def simplex_stripe_tensors(dimensions,  # type: int
                           centers_in,  # type: List[List[int]]
                           centers_out,  # type: List[List[int]]
                           surrounds_in,  # type: List[List[int]]
                           surrounds_out,  # type: List[List[int]]
                           attractor_function=__euclid_function_generator,
                           # type: Callable[[Real], Callable[[Real], Real]]
                           ):
    """ Generates the minimum number of stripe tensors needed to represent all orientations of thin boundaries in
    n-dimensional space.

    :param dimensions: number of dimensions.
    :param centers_in: list of colors added together on points on the edge_orientation_detector.
    :param centers_out: list of colors outputted on points on the edge_orientation_detector.
    :param surrounds_in: list of colors subtracted together on points off the edge_orientation_detector
    :param surrounds_out: list of colors outputted on points off the edge_orientation_detector.
    :param attractor_function: function that takes in the number of dimensions and outputs a function that takes in
            distances and returns positive values for small distances and negative values for large distances.
    :return: a list of tensors for finding all orientations of boundaries.
    """
    return [stripe_tensor(simplex_vector, center_in, center_out, surround_in, surround_out, attractor_function)
            for simplex_vector, center_in, center_out, surround_in, surround_out
            in zip(__axis_coordinates(dimensions), centers_in, centers_out, surrounds_in, surrounds_out)]


def rgb_2d_stripe_tensors(in_channel=(1, 1, 1)):
    """ Finds stripes and outputs colors based on orientation. For use on 2D sensor input only. 3D will require 4 colors
     to visualize.

    :return: a list of tensors for 2D boundary detection and displaying.
    """
    x = 2
    return sum(simplex_stripe_tensors(2, [in_channel, in_channel, in_channel],
                                      [[2 * x, -.5 * x, -.5 * x], [-.5 * x, 2 * x, -.5 * x], [-.5 * x, -.5 * x, 2 * x]],
                                      [in_channel, in_channel, in_channel],
                                      [[-2 * x, .5 * x, .5 * x], [.25 * x, -2 * x, .5 * x], [.5 * x, .5 * x, -2 * x]]))
