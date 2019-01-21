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

    def __init__(self, output_size=(36*4, 24*4), output_colors=3, zoom_ratio=m.e ** .5):
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
        z_tensor = np.asarray([frame], dtype=np.float32)
        #z_tensor = zoom_tensor.from_image(z_tensor, self.output_colors, self.output_size, self.zoom_ratio)
        return z_tensor

    '''def display(self,
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
                result_frames = [result_frame]
            else:
                try:
                    result_frame = np.concatenate((result_frame, result_frame_list), axis=0)
                    result_frames = [result_frame]
                except ValueError:
                    result_frames = [result_frame_list, result_frame]
        return result_frames'''

    def display(self,
                frame,
                cam_id,
                ):
        frame_from_callback = self.callback(frame, cam_id)
        return np.array(frame_from_callback)/255.0

    def run_camera(self, cam=0, fps_limit=60, size=(99999,99999)):
        t = wp.VideoHandlerThread(cam, [self.display] + wp.display_callbacks,
                                  request_size=size,
                                  high_speed=True,
                                  fps_limit=fps_limit)

        t.start()

        ws.SubscriberWindows(window_names=[str(i) for i in range(10)],
                             video_sources=[cam]
                             ).loop()

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

    def __run_on_picture(self, picture):
        picture = self.__convert_pic_to_np(picture)
        if isinstance(picture, (list, tuple)):
            results = []
            for p in picture:
                results.append(self.display(p, 0))
            return results
        return self.display(picture, 0)

    def run_on_pictures(self, pictures, display_results=False, write_results=False):
        if not isinstance(pictures, (list, tuple)):
            return self.__run_on_picture(pictures)
        else:
            results = []
            for picture in pictures:
                results.append(self.__run_on_picture(picture))
            if display_results:
                self.display_picture_results(results)
            if write_results:
                self.write_picture_results(results)
            return results

    def display_picture_results(self, picture, name="pic result"):
        if isinstance(picture, (list, tuple)):
            for p in range(len(picture)):
                self.display_picture_results(picture[p], name + ":{}".format(p))
        else:
            cv2.imshow(name, picture)

    def write_picture_results(self, picture, name="pic result"):
        if isinstance(picture, (list, tuple)):
            for p in range(len(picture)):
                self.write_picture_results(picture[p], name + "-{}".format(p))
        else:
            cv2.imwrite("{}.jpg".format(name), picture)


if __name__ == '__main__':
    filter = PyramidFilter()

    filter.run_camera()

