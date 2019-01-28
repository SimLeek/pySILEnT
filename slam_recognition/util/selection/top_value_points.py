import tensorflow as tf
from tensorflow.python.ops import math_ops, array_ops
from slam_recognition.util.color.get_value import get_value_from_color

if False:
    from typing import Optional


def top_value_points(color_tensor,  # type: tf.Tensor
                     top_percent=0.1,
                     value_tensor=None  # type: Optional[tf.Tensor]
                     ):
    """turns everything except the top percentage of values in a tensor to zero."""
    if value_tensor is None:
        value_tensor = get_value_from_color(color_tensor)

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

def max_value_indices_region(color_tensor,  # type: tf.Tensor
                            region_shape,
                     value_tensor=None  # type: Optional[tf.Tensor]
                     ):
    if value_tensor is None:
        value_tensor = get_value_from_color(color_tensor)

    max_pooled_in_tensor_2 = tf.nn.max_pool(value_tensor, (1, color_tensor.shape[1], color_tensor.shape[2], 1),
                                            strides=(1, region_shape[1], region_shape[2], 1),
                                            padding='SAME')
    resized_pool = tf.image.resize_images(max_pooled_in_tensor_2, color_tensor.shape[1:3],
                                          method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    focus_indices = tf.where(tf.greater_equal(value_tensor, resized_pool))
    return focus_indices

def top_value_points_region(color_tensor,  # type: tf.Tensor
                            region_shape,
                     top_percent=0.1,
                     value_tensor=None  # type: Optional[tf.Tensor]
                     ):
    """turns everything except the top percentage of values in a tensor to zero."""
    if value_tensor is None:
        value_tensor = get_value_from_color(color_tensor)

    pool_shape = tf.shape(color_tensor)/region_shape

    max_pooled_in_tensor_2 = tf.nn.max_pool(value_tensor, (1, color_tensor.shape[1], color_tensor.shape[2], 1),
                                            strides=(1, region_shape[1], region_shape[2], 1),
                                            padding='SAME')
    min_pooled_in_tensor_2 = -1.0 * tf.nn.max_pool(-value_tensor, (1, color_tensor.shape[1], color_tensor.shape[2], 1),
                                                   strides=(1, region_shape[1], region_shape[2], 1),
                                                   padding='SAME')
    top_percent_pool = (1.0 - top_percent) * max_pooled_in_tensor_2 + (top_percent) * min_pooled_in_tensor_2
    resized_pool = tf.image.resize_images(top_percent_pool, color_tensor.shape[1:3],
                                          method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    focus_indices = tf.where(tf.greater_equal(value_tensor, resized_pool))
    focus_sparse = tf.SparseTensor(focus_indices, tf.ones(focus_indices.shape[0]), color_tensor.shape)
    split_sparse = tf.sparse.split(focus_sparse, pool_shape[1], 1)
    focus_panes = tf.split
    vis_max_2 = color_tensor * tf.image.grayscale_to_rgb()

    return vis_max_2
