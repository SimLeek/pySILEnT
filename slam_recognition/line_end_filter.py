from slam_recognition.orientation_filter import OrientationFilter
import tensorflow as tf
from slam_recognition.regulator_tensor.gaussian_regulator_tensor import regulate_tensor, blur_tensor
from slam_recognition.end_tensor import rgb_2d_end_tensors
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops


class LineEndFilter(OrientationFilter):
    callback_depth = 1

    def __init__(self, n_dimensions=2, **argv):
        """Mimics the blob cells in the lowest layer of the V1 in the neocortex, activating pixels that have high
        color difference."""

        super(LineEndFilter, self).__init__(**argv)
        self.tensor_return_type.append(tf.Tensor)

        self.simplex_end_stop = rgb_2d_end_tensors()


    def compile(self, pyramid_tensor):
        super(LineEndFilter, self).compile(pyramid_tensor)

        with tf.name_scope('LineEndFilter Compile') and tf.device('/device:GPU:0'):
            simplex_end_filter = tf.constant(self.simplex_end_stop, dtype=tf.float32, shape=(3, 3, 3, 3))

            compiled_line_end = tf.maximum(
                tf.nn.conv2d(
                    input=self.compiled_list[-1], filter=simplex_end_filter, strides=[1, 1, 1, 1], padding='SAME'),
                [0]
            )

            # todo: move to its own filter class
            '''rgb_weights = [0.3333, 0.3333, 0.3333]
            gray_float = math_ops.tensordot(compiled_line_end, rgb_weights, [-1, -1])
            gray_float = array_ops.expand_dims(gray_float, -1)

            max_pooled_in_tensor = tf.nn.pool(gray_float, window_shape=(5, 5), pooling_type='MAX', padding='SAME')
            max_pooled_in_tensor = tf.image.grayscale_to_rgb(max_pooled_in_tensor)
            gray_float = tf.image.grayscale_to_rgb(gray_float)

            compiled_line_end = compiled_line_end * tf.where(tf.equal(gray_float, max_pooled_in_tensor),
                                                             tf.ones_like(compiled_line_end),
                                                             tf.zeros_like(compiled_line_end))'''

            self.compiled_list.append(tf.clip_by_value(compiled_line_end, 0, 255))


if __name__ == '__main__':
    filter = LineEndFilter()

    # filter.run_camera(cam=r"C:\\Users\\joshm\\Videos\\2017-01-09 23-12-33.mp4", fps_limit=30)
    filter.run_camera(0)
    # results = filter.run_on_pictures([r'C:\Users\joshm\OneDrive\Pictures\backgrounds'], write_results=True)
