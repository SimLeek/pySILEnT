import tensorflow as tf
from cvpubsubs import webcam_pub as w
from cvpubsubs.window_sub import SubscriberWindows
import math as m

from slam_recognition import zoom_tensor
from slam_recognition.center_surround_tensor import rgby_3
from slam_recognition.zoom_tensor.to_image_list import zoom_tensor_to_image_list


class RetinalFilter(object):

    def __init__(self):
        self.rgby = rgby_3(2)

    def spatial_color_2d(self, pyramid_tensor):
        # from: http://thegrimm.net/2017/12/14/tensorflow-image-convolution-edge-detection/

        tf.reset_default_graph()

        input_placeholder = tf.placeholder(
            dtype=tf.float32, shape=(pyramid_tensor.shape))

        with tf.name_scope('center_surround') and tf.device('/device:CPU:0'):  # Use CPU for small filters like these
            conv_w_edge = tf.constant(self.rgby, dtype=tf.float32, shape=(3, 3, 3, 3))

            output_edge = tf.maximum(tf.nn.conv2d(input=input_placeholder, filter=conv_w_edge, strides=[1, 1, 1, 1],
                                                  padding='VALID'), [0])

        with tf.Session() as session:
            result_edge = session.run(output_edge, feed_dict={
                input_placeholder: pyramid_tensor[:, :, :, :]})

        return result_edge

    def callback(self, frame, cam_id):
        frame = zoom_tensor_to_image_list(
            self.spatial_color_2d(zoom_tensor.from_image(frame, 3, [64, 36], m.e ** .5)))
        return frame


if __name__ == '__main__':
    retina = RetinalFilter()

    t = w.VideoHandlerThread(0, [retina.callback] + w.display_callbacks,
                             request_size=(1280, 720),
                             high_speed=True,
                             fps_limit=240
                             )

    t.start()

    SubscriberWindows(window_names=["scale " + str(i) for i in range(6)],
                      video_sources=[0]
                      ).loop()

    w.CamCtrl.stop_cam(0)

    t.join()
