import numpy as np
import slam_recognition.zoom_tensor as zoom_tensor
import math as m
from cvpubsubs import webcam_pub as wp
from cvpubsubs import window_sub as ws
import tensorflow as tf
import cv2
import os

try:
    from PIL import Image as pilImage
except ImportError:
    pass


class PyramidFilter(object):
    callback_depth = 1

    def __init__(self, output_size=(int(36*1), int(24*1)), output_colors=3, zoom_ratio=m.e ** .5):
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
        z_tensor = np.asarray(frame, dtype=np.float32)
        z_tensor = zoom_tensor.from_image(z_tensor, self.output_colors, self.output_size, self.zoom_ratio)
        return z_tensor

    def display(self,
                frame,
                cam_id,
                ):
        frame_from_callback = self.callback(frame, cam_id)
        return [np.array(frame_from_callback[x])/255.0 for x in range(len(frame_from_callback))]

    def run_camera(self, cam=0, fps_limit=60, size=(-1,-1), mjpeg_compression=True):
        t = wp.VideoHandlerThread(video_source=cam, callbacks=[self.display]+wp.display_callbacks,
                                  request_size=size,
                                  high_speed=mjpeg_compression,
                                  fps_limit=fps_limit)

        t.start()

        ws.SubscriberWindows(window_names=[str(i) for i in range(50)],
                             video_sources=[cam],
                             ).loop()
        if isinstance(cam, np.ndarray):
            cam = str(hash(str(cam)))
        wp.CamCtrl.stop_cam(cam)

        t.join()

    @staticmethod
    def __convert_pic_to_np(picture):
        if isinstance(picture, str):
            try:
                if os.path.isfile(picture):
                    cv_result = cv2.imread(picture)
                elif os.path.isdir(picture):
                    cv_result = []
                    for root, subdirs, files in os.walk(picture):
                        for file in files:
                            cv_result.append(cv2.imread(os.path.join(root, file)))
            except:
                pass
            if cv_result is None:
                picture = pilImage.open(picture)
            else:
                picture = cv_result

        try:
            if isinstance(picture, pilImage.Image):
                picture = np.array(picture)
        except:
            pass

        return picture

    def __run_on_picture(self, picture, resize):
        picture = self.__convert_pic_to_np(picture)
        if isinstance(picture, (list, tuple)):
            results = []
            for p in picture:
                results.append(self.run_camera(p, size=resize))
            return results
        return self.run_camera(picture, size=resize)

    def run_on_pictures(self, pictures, resize=(-1,-1)):
        if not isinstance(pictures, (list, tuple)):
            return self.__run_on_picture(pictures, resize)
        else:
            results = []
            for picture in pictures:
                results.append(self.__run_on_picture(picture, resize))
            return results


if __name__ == '__main__':
    filter = PyramidFilter()

    filter.run_camera()

