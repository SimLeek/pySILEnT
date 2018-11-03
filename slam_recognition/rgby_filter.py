from slam_recognition.rgc_filter import RGCFilter
from slam_recognition.center_surround_tensor import rgby_3
import tensorflow as tf

class RGBYFilter(RGCFilter):
    callback_depth = 3

    def __init__(self, n_dimensions=2, **argv):
        """Mimics the blob cells in the lowest layer of the V1 in the neocortex, activating pixels that have high
        color difference."""

        super(RGBYFilter, self).__init__(**argv)

        self.rgby = rgby_3(n_dimensions)

    def compile(self, pyramid_tensor):
        super(RGBYFilter, self).compile(pyramid_tensor)

        with tf.name_scope('RGBYFilter Compile') and tf.device('/device:GPU:0'):
            conv_rgby = tf.constant(self.rgby, dtype=tf.float32, shape=(3, 3, 3, 3))
            compiled_rgby = tf.maximum(tf.nn.conv2d(input=self.compiled_list[-1], filter=conv_rgby, strides=[1, 1, 1, 1],
                                                  padding='SAME'), [0])
            self.compiled_list.append(compiled_rgby)


if __name__ == '__main__':
    filter = RGBYFilter()

    filter.run_camera()
