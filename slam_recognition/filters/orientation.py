import tensorflow as tf
import numpy as np
from slam_recognition.constant_convolutions.edge_orientation_detector import rgb_2d_stripe_tensors
from slam_recognition.util.regulator import regulate_tensor
from slam_recognition.constant_convolutions.gaussian_blur.gaussian_blur import blur_tensor
from slam_recognition.util.get_dimensions import get_dimensions

if False:
    from typing import Union


def orientation_filter(tensor,  # type: Union[tf.Tensor, np.ndarray]
                       blur_size=7
                       ):
    """Mimics the blob cells in the lowest layer of the V1 in the neocortex, activating pixels that have high
            color difference."""
    dimensions = get_dimensions(tensor)

    simplex_boundaries_b = rgb_2d_stripe_tensors()  # todo: make n-d
    blur = blur_tensor(dimensions, lengths=blur_size)

    simplex_orientation_filter_b = tf.constant(simplex_boundaries_b, dtype=tf.float32, shape=(3, 3, 3, 3))

    compiled_orient = tf.maximum(
        tf.nn.conv2d(
            input=tensor, filter=simplex_orientation_filter_b,
            strides=[1, 1, 1, 1], padding='SAME'),
        [0]
    )

    conv_blur = tf.constant(blur, dtype=tf.float32, shape=(blur_size, blur_size, 3, 3))

    compiled_regulated_orient = regulate_tensor(compiled_orient, conv_blur, 1.0, .1)

    return compiled_regulated_orient
