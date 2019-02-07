from slam_recognition.constant_convolutions.center_surround import midget_rgc
from slam_recognition.pyramid_filter import PyramidFilter
import tensorflow as tf
from slam_recognition.util.get_dimensions import get_dimensions


def rgc_filter(tensor  # type: tf.Tensor
               ):
    n_dimensions = get_dimensions(tensor)
    rgc = midget_rgc(n_dimensions)

    conv_rgc = tf.constant(rgc, dtype=tf.float32, shape=(3, 3, 3, 3))

    compiled_rgc = tf.maximum(
        tf.nn.conv2d(input=tensor, filter=conv_rgc, strides=[1, 1, 1, 1], padding='SAME'),
        [0]
    )

    return compiled_rgc
