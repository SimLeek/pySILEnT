import tensorflow as tf
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops


def get_value_from_color(color_tensor  # type: tf.Tensor
                         ):
    """Gets the value from an image tensor, no matter how many colors."""
    color_div = 1.0 / tf.cast(tf.shape(color_tensor)[-1], tf.float32)
    value_float = tf.reduce_sum(color_tensor, -1, keepdims=True)
    value_float = value_float * color_div
    return value_float
