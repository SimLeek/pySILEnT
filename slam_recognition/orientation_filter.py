from slam_recognition.rgby_filter import RGBYFilter
import tensorflow as tf
from slam_recognition.edge_tensor import rgb_2d_stripe_tensors
from slam_recognition.regulator_tensor.gaussian_regulator_tensor import regulate_tensor, blur_tensor


class OrientationFilter(RGBYFilter):
    callback_depth = 4

    def __init__(self, n_dimensions=2, **argv):
        """Mimics the blob cells in the lowest layer of the V1 in the neocortex, activating pixels that have high
        color difference."""

        super(OrientationFilter, self).__init__(**argv)

        self.simplex_boundaries_b = rgb_2d_stripe_tensors()

        self.blur = blur_tensor(2, lengths=7)

    def compile(self, pyramid_tensor):
        super(OrientationFilter, self).compile(pyramid_tensor)

        with tf.name_scope('OrientationFilter Compile') and tf.device('/device:GPU:0'):
            simplex_orientation_filter_b = tf.constant(self.simplex_boundaries_b, dtype=tf.float32, shape=(3, 3, 3, 3))

            compiled_orient = tf.maximum(
                tf.nn.conv2d(
                    input=self.compiled_list[-1], filter=simplex_orientation_filter_b,
                    strides=[1, 1, 1, 1], padding='SAME'),
                [0]
            )

            conv_blur = tf.constant(self.blur, dtype=tf.float32, shape=(7, 7, 3, 3))

            compiled_regulated_orient = regulate_tensor(compiled_orient, conv_blur, 1.0, .1)

            self.compiled_list.append(compiled_regulated_orient)


if __name__ == '__main__':
    filter = OrientationFilter()

    filter.run_camera()
