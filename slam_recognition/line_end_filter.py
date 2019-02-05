from slam_recognition.rgby_filter import RGBYFilter
import tensorflow as tf
from slam_recognition.end_tensor import rgb_2d_end_tensors
from slam_recognition.util.energy.exhaustion import initialize_exhaustion, get_exhaustion

from slam_recognition.util.selection.top_value_points import max_value_indices_region
from slam_recognition.util.color.get_value import get_value_from_color
from slam_recognition.util.selection.isolate_rectangle import pad_inwards
from slam_recognition.util.apply_filter import apply_filter
from slam_recognition.util import get_centroids
from slam_recognition.filters import orientation_filter
import math as m

debug = True


class LineEndFilter(RGBYFilter):
    callback_depth = 2

    def __init__(self, n_dimensions=2, **argv):
        """Mimics the blob cells in the lowest layer of the V1 in the neocortex, activating pixels that have high
        color difference."""

        super(LineEndFilter, self).__init__(**argv)
        self.tensor_return_type.append(tf.Tensor)

        self.simplex_end_stop = rgb_2d_end_tensors()
        self.precompile_list = []

        self.constant_recovery = True
        self.input_based_recovery = False

        self.excitation_max = 8
        self.top_percent_pool = .5
        self.rotation_invariance = False
        self.padded_firing = None
        self.centroid_region_shape = [1, 3, 3]  # 2 or 3 are good values for this
        self.region_shape = [1, self.output_size[1] / 2.0, self.output_size[0] / 2.0, self.output_colors]

    def pre_compile(self, pyramid_tensor):
        self.input_placeholder = tf.placeholder(dtype=tf.float32, shape=(pyramid_tensor.shape))

        line_end_tensor = tf.maximum(apply_filter(self.input_placeholder, self.simplex_end_stop), [0])
        line_end_tensor = tf.clip_by_value(line_end_tensor, 0, 255.0)
        padded_line_end_tensor = pad_inwards(line_end_tensor, [[0, 0], [2, 2], [2, 2], [0, 0]])

        gray_line_end_tensor = get_value_from_color(padded_line_end_tensor)

        centroids, centroid_importances = get_centroids(gray_line_end_tensor / 255.0, self.centroid_region_shape,
                                                        debug=True)
        self.energy_values = initialize_exhaustion(centroid_importances * 255)

        self.precompile_list = [self.energy_values]

    def compile(self, pyramid_tensor):
        with tf.name_scope('LineEndFilter Compile') and tf.device('/device:GPU:0'):
            super(LineEndFilter, self).compile(pyramid_tensor)

            if isinstance(self.region_shape, list):
                self.region_shape = tf.TensorShape(self.region_shape)

            orient_tensor = orientation_filter(self.compiled_list[-1])

            line_end_tensor = tf.maximum(apply_filter(orient_tensor, self.simplex_end_stop), [0])
            line_end_tensor = tf.clip_by_value(line_end_tensor, 0, 255)
            padded_line_end_tensor = pad_inwards(line_end_tensor, [[0, 0], [2, 2], [2, 2], [0, 0]])

            gray_line_end_tensor = get_value_from_color(padded_line_end_tensor)

            centroids, centroid_importances = get_centroids(gray_line_end_tensor / 255.0, self.centroid_region_shape,
                                                            debug=True)
            centroid_importances = tf.clip_by_value(centroid_importances * (255 / 4.0), 1, 256) - 1
            half_shape = tf.cast(gray_line_end_tensor.shape[1:3], tf.float32) / tf.constant(m.e ** .5)
            im2 = tf.image.resize_nearest_neighbor(gray_line_end_tensor, tf.cast(half_shape, tf.int32))
            centroids2, centroid_importances2 = get_centroids(im2 / 255.0, self.centroid_region_shape, debug=True)

            fired_importants, update_importances = get_exhaustion(centroid_importances, self.energy_values,
                                                                  for_visualizing=True)
            # fired_importants2, update_importances2 = get_exhaustion(centroid_importances2, self.energy_values, for_visualizing=True)

            top_percent_points = max_value_indices_region(padded_line_end_tensor,
                                                          self.region_shape, gray_line_end_tensor)

            # top_idxs = tf.cast(top_percent_points, tf.int32)

        # with tf.name_scope('LineEndFilter Compile pt2') and tf.device('/device:CPU:0'):
        #    relativity_tensor = get_relative_to_indices_regions(has_fired2,self.region_shape, top_percent_points)

        self.compiled_list.extend(
            [255 - centroids * 255, 255 - centroids2 * 255, fired_importants * 255, update_importances,
             padded_line_end_tensor])
        # self.compiled_list.extend(
        #    [tf.image.grayscale_to_rgb(update_energy * (255.0 / 16) + 127.5), centroids,
        #     relativity_tensor, padded_line_end_tensor])
        # self.compiled_list.extend([tf.clip_by_value(compiled_line_end_0, 0, 255)])

    def run(self, pyramid_tensor):

        if self.pyramid_tensor_shape != pyramid_tensor.shape:
            g = tf.Graph()
            with g.as_default():
                self.pyramid_tensor_shape = pyramid_tensor.shape
                self.pre_compile(pyramid_tensor)
                initter = tf.initializers.global_variables()
                self.compile(pyramid_tensor)
                feed_dict = dict({self.input_placeholder: pyramid_tensor[:, :, :, :]})
                self.session = tf.Session()
                self.session.run(initter)
                # self.session.run(self.precompile_list, feed_dict=feed_dict)

        if self.session is None:
            g = tf.Graph()
            with g.as_default():
                self.pre_compile(pyramid_tensor)
                initter = tf.initializers.global_variables()
                self.compile(pyramid_tensor)
                self.session = tf.Session()
                feed_dict = dict({self.input_placeholder: pyramid_tensor[:, :, :, :]})
                self.session.run(initter)
                # self.session.run(self.precompile_list, feed_dict=feed_dict)
        feed_dict = dict({self.input_placeholder: pyramid_tensor[:, :, :, :]})

        result = self.session.run(self.compiled_list[3:8], feed_dict=feed_dict)

        return result

    def callback(self,
                 frame,
                 cam_id=None,
                 depth=2
                 ):
        z_tensor = super(LineEndFilter, self).callback(frame, cam_id)
        tensors = self.run(z_tensor)
        return [frame] + [[tensors[x][y] for y in range(len(tensors[x]))] for x in range(4)]


if __name__ == '__main__':
    filter = LineEndFilter()

    # filter.run_camera(cam=r"C:\\Users\\joshm\\Videos\\2019-01-18 21-49-54.mp4")
    filter.run_on_pictures(r"C:\\Users\\joshm\\OneDrive\\Pictures\\robots\\repr\\phone.png", resize=(-1, 480))
    # filter.run_camera(0, size=(800, 600))
    # results = filter.run_on_pictures([r'C:\Users\joshm\OneDrive\Pictures\backgrounds'], write_results=True)
