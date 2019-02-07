import tensorflow as tf
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops


def get_bw_from_color(color_tensor  # type: tf.Tensor
                      ):
    """Gets the value from an image tensor, no matter how many colors."""
    color_weights = tf.tile([1.0], [tf.shape(color_tensor)[-1]])
    value_float = math_ops.tensordot(color_tensor, color_weights, [-1, -1])
    value_float = array_ops.expand_dims(value_float, -1)
    bw = tf.where(tf.not_equal(value_float, 0), tf.ones_like(value_float), tf.zeros_like(value_float))
    return bw
