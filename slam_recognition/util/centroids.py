from slam_recognition.util import index_tensor
import tensorflow as tf
from tensorflow.python.ops import array_ops
from slam_recognition.util import color
from slam_recognition.util import math
import itertools
import numpy as np

def additive_filter(shape, channels):
    filter_out = np.zeros(shape+[channels,channels])
    channel_io = np.zeros([channels,channels])
    for i in range(channels):
        ch = np.zeros([channels])
        ch[i] = 1
        channel_io[i] = ch
    for xyz in itertools.product([range(x) for x in shape]):
        filter_out[xyz] = np.asarray([[1,0],[0,1]])
    return tf.constant(filter_out, dtype=tf.float32)


def get_centroids(value_tensor,  # type: tf.Tensor
                  region_shape,
                  debug=False
                  ):
    ind_tens = index_tensor.from_tensor(value_tensor)
    ind_tens = tf.cast(ind_tens, tf.float32)
    #if debug:
    #    with tf.control_dependencies([tf.assert_greater_equal(value_tensor, tf.zeros_like(value_tensor)),
    #                                  tf.assert_less_equal(value_tensor, tf.ones_like(value_tensor))]):
    #        tf.identity(value_tensor)
    num_channels = len(ind_tens.shape.as_list())-1
    full_channel_values = color.to_channels(value_tensor, num_channels)
    biased_index_tensor = ind_tens*full_channel_values
    centroid_pool = tf.nn.convolution(biased_index_tensor,
                                      additive_filter(region_shape[1:], num_channels),
                                      strides=region_shape[1:],
                                      padding="SAME")
    total_pool = tf.nn.convolution(full_channel_values,
                                   tf.ones(region_shape[1:]+[num_channels]+[1])/num_channels,
                                   strides=region_shape[1:],
                                   padding="SAME")
    corrected_centroid_pool = centroid_pool/total_pool
    resized_pool = tf.image.resize_nearest_neighbor(corrected_centroid_pool, value_tensor.shape[1:num_channels+1])
    centroids = math.equality_distance(resized_pool, ind_tens)
    value_centroids = tf.reduce_sum(centroids, -1, keepdims=True)
    return value_centroids

def get_centroids_array(value_tensor,  # type: tf.Tensor
                  region_shape,
                  debug=False
                  ):
    ind_tens = index_tensor.from_tensor(value_tensor)
    ind_tens = tf.cast(ind_tens, tf.float32)
    #if debug:
    #    with tf.control_dependencies([tf.assert_greater_equal(value_tensor, tf.zeros_like(value_tensor)),
    #                                  tf.assert_less_equal(value_tensor, tf.ones_like(value_tensor))]):
    #        tf.identity(value_tensor)
    num_channels = len(ind_tens.shape.as_list())-1
    full_channel_values = color.to_channels(value_tensor, num_channels)
    biased_index_tensor = ind_tens*full_channel_values
    centroid_pool = tf.nn.convolution(biased_index_tensor,
                                      additive_filter(region_shape[1:], num_channels),
                                      strides=region_shape[1:],
                                      padding="SAME")
    total_pool = tf.nn.convolution(full_channel_values,
                                   tf.ones(region_shape[1:]+[num_channels]+[1])/num_channels,
                                   strides=region_shape[1:],
                                   padding="SAME")
    corrected_centroid_pool = centroid_pool/total_pool
    return corrected_centroid_pool

