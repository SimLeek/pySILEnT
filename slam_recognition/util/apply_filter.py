import tensorflow as tf


def apply_filter(tensor, filter):
    filter_constant = tf.constant(filter, dtype=tf.float32)
    filtered_tensor = tf.nn.conv2d(input=tensor, filter=filter_constant, strides=[1, 1, 1, 1], padding='SAME')
    return filtered_tensor
