import tensorflow as tf
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops


def get_value_from_color(color_tensor  # type: tf.Tensor
                         ):
    """Gets the value from an image tensor, no matter how many colors."""
    color_div = [1.0 / tf.cast(tf.shape(color_tensor)[-1], tf.float32)]
    color_weights = tf.tile(color_div, [tf.shape(color_tensor)[-1]])
    value_float = math_ops.tensordot(color_tensor, color_weights, [-1, -1])
    value_float = array_ops.expand_dims(value_float, -1)
    return value_float
