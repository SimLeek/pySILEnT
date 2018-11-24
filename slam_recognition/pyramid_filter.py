import numpy as np
import slam_recognition.zoom_tensor as zoom_tensor
import math as m
from cvpubsubs import webcam_pub as wp
from cvpubsubs import window_sub as ws
import tensorflow as tf


class PyramidFilter(object):
    callback_depth = 1

    def __init__(self, output_size=(64, 32), output_colors=3, zoom_ratio=m.e ** .5):
        """Generates several smaller images at different zoom levels from one input image."""
        self.output_size = output_size
        self.output_colors = output_colors
        self.zoom_ratio = zoom_ratio

        self.tensor_return_type = []
        self.tensor_return_type.append(tf.Tensor)

    def callback(self,
                 frame,
                 cam_id,
                 depth=callback_depth
                 ):
        """Transforms the input frame into an image pyramid.

        :param frame: Input frame.
        :param cam_id: Unused.
        :return: Zoom tensor to use.
        """
        np_frame = np.asarray(frame, dtype=np.float32)
        z_tensor = zoom_tensor.from_image(np_frame, self.output_colors, self.output_size, self.zoom_ratio)
        if depth >= 1:
            return z_tensor
        else:
            return None

    def display(self,
                frame,
                cam_id,
                ):
        frame_from_callback = self.callback(frame, cam_id)
        if frame_from_callback is None:
            return None
        frame_array = [zoom_tensor.to_image_list(f) for f in frame_from_callback]
        if len(frame_array) == 0:
            return
        result_frame = None
        for frame_list in frame_array:
            for f in range(len(frame_list)):
                if frame_list[f].ndim == 2:
                    frame_list[f] = np.stack((frame_list[f],) * 3, axis=-1)
            result_frame_list = frame_list[0]
            for f in range(1, len(frame_list)):
                result_frame_list = np.concatenate((result_frame_list, frame_list[f]), axis=1)
            if result_frame is None:
                result_frame = result_frame_list
            else:
                try:
                    result_frame = np.concatenate((result_frame, result_frame_list), axis=0)
                    result_frames = [result_frame]
                except ValueError:
                    result_frames = [result_frame_list, result_frame]
        return result_frames

    def run_camera(self):
        t = wp.VideoHandlerThread(0, [self.display] + wp.display_callbacks,
                                  request_size=(99999, 99999),
                                  high_speed=True,
                                  fps_limit=240)

        t.start()

        ws.SubscriberWindows(window_names=["Pyramid", "t"],
                             video_sources=[0]
                             ).loop()

        wp.CamCtrl.stop_cam(0)

        t.join()


if __name__ == '__main__':
    filter = PyramidFilter()

    filter.run_camera()
