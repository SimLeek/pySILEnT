import tensorflow as tf
from tensorflow.python.ops import math_ops, array_ops

if False:
    from typing import Optional


def top_value_points(color_tensor,  # type: tf.Tensor
                     top_percent=0.1,
                     value_tensor=None  # type: Optional[tf.Tensor]
                     ):
    """turns everything except the top percentage of values in a tensor to zero."""
    if value_tensor is None:
        rgb_weights = [0.3333, 0.3333, 0.3333]
        value_tensor = math_ops.tensordot(color_tensor, rgb_weights, [-1, -1])
        value_tensor = array_ops.expand_dims(value_tensor, -1)

    max_pooled_in_tensor_2 = tf.nn.max_pool(value_tensor, (1, color_tensor.shape[1], color_tensor.shape[2], 1),
                                            strides=(1, color_tensor.shape[1], color_tensor.shape[2], 1),
                                            padding='SAME')
    min_pooled_in_tensor_2 = -1.0 * tf.nn.max_pool(-value_tensor, (1, color_tensor.shape[1], color_tensor.shape[2], 1),
                                                   strides=(1, color_tensor.shape[1], color_tensor.shape[2], 1),
                                                   padding='SAME')
    top_percent_pool = (1.0 - top_percent) * max_pooled_in_tensor_2 + (top_percent) * min_pooled_in_tensor_2
    resized_pool = tf.image.resize_images(top_percent_pool, value_tensor.shape[1:3],
                                          method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    vis_max_2 = color_tensor * tf.image.grayscale_to_rgb(tf.where(tf.greater_equal(value_tensor, resized_pool),
                                                                  tf.ones_like(resized_pool),
                                                                  tf.zeros_like(resized_pool)))

    return vis_max_2
