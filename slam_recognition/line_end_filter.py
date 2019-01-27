from slam_recognition.orientation_filter import OrientationFilter
import tensorflow as tf
from slam_recognition.end_tensor import rgb_2d_end_tensors

from slam_recognition.util.energy.recovery import generate_recovery
from slam_recognition.util.selection.top_value_points import top_value_points
from slam_recognition.util.color.get_value import get_value_from_color
from slam_recognition.util.selection.isolate_rectangle import pad_inwards
from slam_recognition.util.relativity.get_relative_to_indices import get_relative_to_indices

class LineEndFilter(OrientationFilter):
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
        self.__pool_width = 0
        self.__pool_height = 0
        self.top_percent_pool = 0.4
        self.rotation_invariance = False

        self.relativity_tensor_shape = [4, self.output_size[1] * 2, self.output_size[0] * 2, 3]

    def pre_compile(self, pyramid_tensor):
        self.input_placeholder = tf.placeholder(dtype=tf.float32, shape=(pyramid_tensor.shape))

        gray_float = get_value_from_color(self.input_placeholder)
        self.energy_values = tf.Variable(tf.ones_like(gray_float) * self.excitation_max, dtype=tf.float32)

        self.precompile_list = [self.energy_values]

        self.padded_firing = None

    def compile(self, pyramid_tensor):
        with tf.name_scope('LineEndFilter Compile') and tf.device('/device:CPU:0'):
            super(LineEndFilter, self).compile(pyramid_tensor)

            simplex_end_filter = tf.constant(self.simplex_end_stop, dtype=tf.float32, shape=(3, 3, 3, 3))

            compiled_line_end_0 = tf.maximum(
                tf.nn.conv2d(
                    input=self.compiled_list[-1], filter=simplex_end_filter, strides=[1, 1, 1, 1], padding='SAME'),
                [0]
            )

            compiled_line_end_0 = pad_inwards(compiled_line_end_0, [[0, 0], [2, 2], [2, 2], [0, 0]])

            gray_float = get_value_from_color(compiled_line_end_0)

            # todo: replace with faster and more stable grouping areas

            memory_values = gray_float ** self.energy_values.value()
            max_pooled_in_tensor = tf.nn.max_pool(memory_values, (1, 3, 3, 1), strides=(1, 1, 1, 1), padding='SAME')

            has_fired = tf.where(tf.logical_and(tf.greater_equal(max_pooled_in_tensor, 100),
                                                tf.equal(memory_values, max_pooled_in_tensor)),
                                 tf.ones_like(max_pooled_in_tensor),
                                 tf.zeros_like(max_pooled_in_tensor))

            fire_strength = has_fired * gray_float

            exhaustion_max = 8

            exhaustion = has_fired * 255.0 - fire_strength

            recovery = generate_recovery(fire_strength, self.input_based_recovery, self.constant_recovery)

            update_energy = self.energy_values.assign(
                tf.clip_by_value((self.energy_values * 255 - exhaustion + recovery) / 255, -exhaustion_max,
                                 self.excitation_max)
            )

            has_fired2 = tf.image.grayscale_to_rgb(has_fired) * compiled_line_end_0

            top_percent_points = top_value_points(has_fired2, self.top_percent_pool, gray_float)

            top_idxs = tf.cast(tf.where(tf.not_equal(top_percent_points, 0)), tf.int32)

            relativity_tensor = get_relative_to_indices(has_fired2, top_idxs)
            #    return i, tf.clip_by_value(added_relative,0,255), slices

            # i_start = tf.constant(0, dtype=tf.int64)
            # i, relativity_tensor, idxs = tf.while_loop(body, while_condition, [i_start, relativity_tensor, top_idxs],
            #                                           shape_invariants=[i_start.get_shape(),
            #                                                             tf.TensorShape(self.relativity_tensor_shape),
            #                                                             tf.TensorShape([None,None])]
            #                                           )

            self.compiled_list.extend(
                [tf.image.grayscale_to_rgb(update_energy * (255.0 / 16) + 127.5), has_fired2, top_percent_points,
                 relativity_tensor, compiled_line_end_0])
            # self.compiled_list.extend([tf.clip_by_value(compiled_line_end_0, 0, 255)])

    def run(self, pyramid_tensor):

        if self.pyramid_tensor_shape != pyramid_tensor.shape:
            g = tf.Graph()
            with g.as_default():
                self.__pool_width = pyramid_tensor.shape[1]
                self.__pool_height = pyramid_tensor.shape[2]
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
                self.__pool_width = pyramid_tensor.shape[1]
                self.__pool_height = pyramid_tensor.shape[2]
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
        return [frame] + [[tensors[x][y] for y in range(len(tensors[x]))] for x in range(5)]


if __name__ == '__main__':
    filter = LineEndFilter()

    # filter.run_camera(cam=r"C:\\Users\\joshm\\Videos\\2019-01-18 21-49-54.mp4")
    # filter.run_on_pictures(r"C:\\Users\\joshm\\OneDrive\\Pictures\\robots\\repr\\phone.png", resize=(-1,480))
    filter.run_camera(0, size=(800, 600))
    # results = filter.run_on_pictures([r'C:\Users\joshm\OneDrive\Pictures\backgrounds'], write_results=True)
