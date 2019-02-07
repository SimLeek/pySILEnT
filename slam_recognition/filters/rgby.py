from slam_recognition.constant_convolutions.center_surround import rgby_3
from slam_recognition.util.get_dimensions import get_dimensions
import tensorflow as tf


def rgby_filter(tensor  # type: tf.Tensor
                ):
    n_dimensions = get_dimensions(tensor)
    rgby = rgby_3(n_dimensions)
    conv_rgby = tf.constant(rgby, dtype=tf.float32, shape=(3, 3, 3, 3))
    compiled_rgby = tf.maximum(tf.nn.conv2d(input=tensor, filter=conv_rgby, strides=[1, 1, 1, 1],
                                            padding='SAME'), [0])

    return compiled_rgby
