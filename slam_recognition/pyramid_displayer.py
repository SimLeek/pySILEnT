import numpy as np
import math as m
from cvpubsubs import webcam_pub as wp
from cvpubsubs import window_sub as ws
import tensorflow as tf
import cv2
import os
from abc import abstractmethod, ABCMeta

if False:
    from typing import Union

try:
    from PIL import Image as pilImage
except ImportError:
    pilImage = None


class PyramidDisplayer(object):
    __metaclass__ = ABCMeta

    def __init__(self, output_size=(int(36 * 8), int(24 * 8)), output_colors=3, zoom_ratio=m.e ** .5):
        """Generates several smaller images at different zoom levels from one input image."""
        self.output_size = output_size
        self.output_colors = output_colors
        self.zoom_ratio = zoom_ratio

        self.tensor_return_type = []
        self.tensor_return_type.append(tf.Tensor)

    @abstractmethod
    def callback(self, frame, cam_id):  # NOSONAR
        return [frame]

    def display(self,
                frame,
                cam_id,
                ):
        frame_from_callback = self.callback(frame, cam_id)
        return [np.array(frame_from_callback[x]) / 255.0 for x in range(len(frame_from_callback))]

    def run_camera(self,
                   cam=0,  # type: Union[int, np.ndarray, pilImage.Image]
                   fps_limit=60, size=(-1, -1), mjpeg_compression=True):
        t = wp.VideoHandlerThread(video_source=cam, callbacks=[self.display] + wp.display_callbacks,
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
    def __get_all_picture_files(directory_or_filename):
        try:
            if os.path.isfile(directory_or_filename):
                cv_result = cv2.imread(directory_or_filename)
            elif os.path.isdir(directory_or_filename):
                cv_result = []
                for root, subdirs, files in os.walk(directory_or_filename):
                    for file in files:
                        cv_result.append(cv2.imread(os.path.join(root, file)))
        except:
            cv_result = None
        return cv_result

    @staticmethod
    def __convert_pic_to_np(picture):
        if isinstance(picture, str):
            cv_result = PyramidDisplayer.__get_all_picture_files(picture)
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

    def run_pictures(self, pictures, resize=(-1, -1)):
        if not isinstance(pictures, (list, tuple)):
            return self.__run_on_picture(pictures, resize)
        else:
            results = []
            for picture in pictures:
                results.append(self.__run_on_picture(picture, resize))
            return results


if __name__ == '__main__':
    filter = PyramidDisplayer()

    filter.run_camera()
