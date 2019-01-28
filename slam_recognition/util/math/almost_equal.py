import tensorflow as tf


def almost_equal(tensor1, tensor2, diff=0.51):
    return tf.math.less_equal(tensor1 - tensor2 + diff, diff * 2)


def equality_distance(tensor1, tensor2):
    return tf.abs(tensor1 - tensor2)
