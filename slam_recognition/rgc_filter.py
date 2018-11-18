from slam_recognition.center_surround_tensor import midget_rgc
from slam_recognition.pyramid_filter import PyramidFilter
import tensorflow as tf


class RGCFilter(PyramidFilter):
    callback_depth = 2

    def __init__(self, n_dimensions=2, **argv):
        """Mimics the Retinal Ganglian Cells in the human eye, activating pixels more when central colors are brighter
        than neighboring colors."""

        super(RGCFilter, self).__init__(**argv)
        self.rgc = midget_rgc(n_dimensions)

        self.pyramid_tensor_shape = None
        self.input_placeholder = None
        self.compiled_list = []
        self.session = None

    def compile(self, pyramid_tensor):
        """runs the RGC filter on the set of images."""
        self.compiled_list = []
        tf.reset_default_graph()

        input_placeholder = tf.placeholder(dtype=tf.float32, shape=(pyramid_tensor.shape))

        self.input_placeholder = input_placeholder

        with tf.name_scope('RGCFilter Compile') and tf.device('/device:GPU:0'):
            conv_rgc = tf.constant(self.rgc, dtype=tf.float32, shape=(3, 3, 3, 3))

            compiled_rgc = tf.maximum(
                tf.nn.conv2d(input=input_placeholder, filter=conv_rgc, strides=[1, 1, 1, 1], padding='SAME'),
                [0]
            )

            self.compiled_list.append(compiled_rgc)

    def run(self, pyramid_tensor):
        if self.pyramid_tensor_shape != pyramid_tensor.shape:
            self.pyramid_tensor_shape = pyramid_tensor.shape
            self.compile(pyramid_tensor)
        if self.session is None:
            self.session = tf.Session()
        result = self.session.run(self.compiled_list, feed_dict={
            self.input_placeholder: pyramid_tensor[:, :, :, :]
        })

        return result

    def callback(self,
                 frame,
                 cam_id,
                 depth=None
                 ):
        z_tensor = super(RGCFilter, self).callback(frame, cam_id)
        tensors = self.run(z_tensor)
        result = []
        if self.callback_depth > len(tensors):
            result.append(z_tensor)
        for i in range(len(tensors)):
            if self.callback_depth >= i:
                result.append(tensors[i])
        return result


if __name__ == '__main__':
    filter = RGCFilter()

    filter.run_camera()
