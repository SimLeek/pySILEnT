from slam_recognition.line_end_filter import LineEndFilter
import tensorflow as tf
from slam_recognition.center_surround_tensor.rgc import midget_rgc_full
from slam_recognition.edge_tensor.edge_tensor import rgb_2d_edge_tensors_time_diff
from slam_recognition.end_tensor import rgb_2d_end_tensors_time

from slam_recognition.regulator_tensor.gaussian_regulator_tensor import blur_tensor

class MotionFilter1(LineEndFilter):
    callback_depth = 6

    def __init__(self, n_dimensions=2, **argv):
        super(MotionFilter1, self).__init__(**argv)
        self.tensor_return_type.append(tf.Tensor)

        self.motion_detector = rgb_2d_end_tensors_time()
        self.prev_in_placeholder=None
        self.prev_in=None
        self.prev_out_placeholder = None
        self.prev_out=None

        self.blur_tensor = blur_tensor(2)


    def compile(self, pyramid_tensor):
        super(MotionFilter1, self).compile(pyramid_tensor)

        self.prev_in_placeholder = tf.placeholder_with_default(tf.zeros_like(pyramid_tensor, dtype=tf.float32),
                                                               shape=(pyramid_tensor.shape))
        self.prev_out_placeholder = tf.placeholder_with_default(tf.zeros_like(pyramid_tensor, dtype=tf.float32),
                                                               shape=(pyramid_tensor.shape))

        with tf.name_scope('MotionFilter1 Compile') and tf.device('/device:GPU:0'):
            conv_motion = tf.constant(self.motion_detector, dtype=tf.float32, shape=(7,7,3,3))
            conv_blur = tf.constant(self.blur_tensor, dtype=tf.float32, shape=(3, 3, 3, 3))

            # compute if a point has moved, and create a positive output at the new location if so, and negative otherwise

            prev = self.prev_in_placeholder
            current = self.compiled_list[-1]

            sparkliness =  tf.maximum(
                tf.nn.conv2d(current-prev,filter=conv_motion, strides=[1, 1, 1, 1], padding='SAME'),
                [0]
            )
            #sparkliness = current-prev
            #sparkliness -= 255
            sparkliness = tf.maximum(sparkliness, [0])

            self.compiled_list.append(sparkliness)

    def run(self, pyramid_tensor):
        if self.pyramid_tensor_shape != pyramid_tensor.shape:
            self.pyramid_tensor_shape = pyramid_tensor.shape
            self.compile(pyramid_tensor)
            self.session = tf.Session()
        if self.session is None:
            self.session = tf.Session()
        feed_dict = dict({self.input_placeholder: pyramid_tensor[:, :, :, :]})
        if self.prev_in is not None:
            feed_dict.update({self.prev_in_placeholder: self.prev_in[:, :, :, :]})
        if self.prev_out is not None:
            feed_dict.update({self.prev_out_placeholder: self.prev_out[:, :, :, :]})
        result = self.session.run(self.compiled_list, feed_dict=feed_dict)
        self.prev_in = result[-2]
        self.prev_out = result[-1]
        return result

if __name__ == '__main__':
    filter = MotionFilter1()

    #filter.run_camera(cam=r"C:\\Users\\joshm\\Videos\\TimelineHex.mov", fps_limit=24)
    filter.run_camera()
