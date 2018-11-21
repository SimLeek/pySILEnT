from slam_recognition.orientation_filter import OrientationFilter
import tensorflow as tf
from slam_recognition.regulator_tensor.gaussian_regulator_tensor import regulate_tensor, blur_tensor
from slam_recognition.end_tensor import rgb_2d_end_tensors


class LineEndFilter(OrientationFilter):
    callback_depth = 5

    def __init__(self, n_dimensions=2, **argv):
        """Mimics the blob cells in the lowest layer of the V1 in the neocortex, activating pixels that have high
        color difference."""

        super(LineEndFilter, self).__init__(**argv)
        self.tensor_return_type.append(tf.Tensor)

        self.simplex_end_stop = rgb_2d_end_tensors()

        self.blur = blur_tensor(2, lengths=7)

    def compile(self, pyramid_tensor):
        super(LineEndFilter, self).compile(pyramid_tensor)

        with tf.name_scope('LineEndFilter Compile') and tf.device('/device:GPU:0'):
            simplex_end_filter = tf.constant(self.simplex_end_stop, dtype=tf.float32, shape=(7, 7, 3, 3))

            compiled_line_end = tf.maximum(
                tf.nn.conv2d(
                    input=self.compiled_list[-1], filter=simplex_end_filter, strides=[1, 1, 1, 1], padding='SAME'),
                [0]
            )

            conv_blur = tf.constant(self.blur, dtype=tf.float32, shape=(7, 7, 3, 3))

            compiled_line_end = regulate_tensor(compiled_line_end, conv_blur, 0.5, 0.1)

            max_pooled_in_tensor = tf.nn.pool(compiled_line_end, window_shape=(3, 3), pooling_type='MAX', padding='SAME')
            compiled_line_end = compiled_line_end * tf.where(tf.equal(compiled_line_end, max_pooled_in_tensor), compiled_line_end,
                                              tf.zeros_like(compiled_line_end))

            self.compiled_list.append(compiled_line_end)


if __name__ == '__main__':
    filter = LineEndFilter()

    filter.run_camera()
