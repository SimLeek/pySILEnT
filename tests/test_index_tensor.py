from slam_recognition.util import index_tensor
import tensorflow as tf
from unittest import TestCase


class TestIndexTensor(TestCase):
    def test_rgb_index_array_from_tensor_shape(self):
        in_shape = tf.TensorShape([4, 2, 2, 3])

        index_array = [[[0, 0], [1, 0]],
                       [[0, 1], [1, 1]]]

        test_tensor = index_array
        i_tensor = index_tensor.from_shape(in_shape)

        with tf.control_dependencies([tf.assert_equal(i_tensor, test_tensor)]):
            tf.identity(i_tensor)

    def test_rgb_index_array_from_tensor(self):
        in_shape = tf.TensorShape([4, 2, 2, 3])
        in_tensor = tf.ones(in_shape)

        index_array = [[[0, 0], [1, 0]],
                       [[0, 1], [1, 1]]]

        test_tensor = index_array
        i_tensor = index_tensor.from_tensor(in_tensor)

        with tf.control_dependencies([tf.assert_equal(i_tensor, test_tensor)]):
            tf.identity(i_tensor)
