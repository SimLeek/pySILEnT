from slam_recognition.orientation_filter import OrientationFilter
import tensorflow as tf
from slam_recognition.regulator_tensor.gaussian_regulator_tensor import regulate_tensor, blur_tensor
from slam_recognition.end_tensor import rgb_2d_end_tensors
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from slam_recognition.color_tensor import color_tensor

def generate_constant_recovery(tensor_in, recovery_amount=6):
    return tf.ones_like(tensor_in)*recovery_amount

def generate_input_based_recovery(tensor_in, recovery_percentage=0.8):
    return tensor_in*recovery_percentage

def generate_recovery(tensor_in, is_input_based=False, is_constant=True):
    if is_input_based and not is_constant:
        recovery = generate_input_based_recovery(tensor_in)
    elif is_constant and not is_input_based:
        recovery = generate_constant_recovery(tensor_in)
    elif is_input_based and is_constant:
        recovery = tf.maximum(generate_input_based_recovery(tensor_in), generate_constant_recovery(tensor_in))
    elif not is_input_based and not is_constant:
        raise ValueError("You must choose a type of recovery")
    return recovery

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

    def pre_compile(self, pyramid_tensor):
        self.input_placeholder = tf.placeholder(dtype=tf.float32, shape=(pyramid_tensor.shape))


        rgb_weights = [0.3333, 0.3333, 0.3333]
        gray_float = math_ops.tensordot(self.input_placeholder, rgb_weights, [-1, -1])
        gray_float = array_ops.expand_dims(gray_float, -1)
        self.energy_values = tf.Variable(tf.ones_like(gray_float) , dtype=tf.float32)

        self.precompile_list= [self.energy_values]

    def compile(self, pyramid_tensor):
        with tf.name_scope('LineEndFilter Compile') and tf.device('/device:GPU:0'):
            super(LineEndFilter, self).compile(pyramid_tensor)

            simplex_end_filter = tf.constant(self.simplex_end_stop, dtype=tf.float32, shape=(3, 3, 3, 3))

            compiled_line_end_0 = tf.maximum(
                tf.nn.conv2d(
                    input=self.compiled_list[-1], filter=simplex_end_filter, strides=[1, 1, 1, 1], padding='SAME'),
                [0]
            )

            # todo: move to its own filter class
            rgb_weights = [0.3333, 0.3333, 0.3333]
            gray_float = math_ops.tensordot(compiled_line_end_0, rgb_weights, [-1, -1])
            gray_float = array_ops.expand_dims(gray_float, -1)

            # todo: replace with faster and more stable grouping areas
            max_pooled_in_tensor_2 = tf.nn.max_pool(gray_float, (1, 100, 100, 1), strides=(1, 100, 100, 1),
                                                    padding='SAME')
            resized_pool = tf.image.resize_images(max_pooled_in_tensor_2, gray_float.shape[1:3],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            vis_max_2 = compiled_line_end_0*tf.image.grayscale_to_rgb(tf.where(tf.equal(gray_float, resized_pool),
                                tf.ones_like(resized_pool),
                                                            tf.zeros_like(resized_pool)))

            memory_values = gray_float**self.energy_values.value()
            max_pooled_in_tensor = tf.nn.max_pool(memory_values, (1,3, 3,1), strides=(1,1,1,1), padding='SAME')

            has_fired = tf.where(tf.logical_and(tf.greater_equal(max_pooled_in_tensor,100) ,tf.equal(memory_values, max_pooled_in_tensor)),
                                                             tf.ones_like(max_pooled_in_tensor),
                                                             tf.zeros_like(max_pooled_in_tensor))

            fire_strength = has_fired * gray_float

            cost_of_firing = 200

            exhaustion_max = 8
            excitation_max = 8

            exhaustion = has_fired*cost_of_firing

            recovery = generate_recovery(fire_strength, self.input_based_recovery, self.constant_recovery)

            update_energy = self.energy_values.assign(
                tf.clip_by_value((self.energy_values*255-exhaustion+recovery)/255, -exhaustion_max, excitation_max)
            )

            has_fired2 = tf.image.grayscale_to_rgb(has_fired) * compiled_line_end_0




            self.compiled_list.extend([tf.image.grayscale_to_rgb(update_energy*(255.0/16)+127.5),vis_max_2,tf.clip_by_value(has_fired2, 0, 255), compiled_line_end_0])
            #self.compiled_list.extend([tf.clip_by_value(compiled_line_end_0, 0, 255)])

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
                #self.session.run(self.precompile_list, feed_dict=feed_dict)

        if self.session is None:
            g = tf.Graph()
            with g.as_default():
                self.pre_compile(pyramid_tensor)
                initter = tf.initializers.global_variables()
                self.compile(pyramid_tensor)
                self.session = tf.Session()
                feed_dict = dict({self.input_placeholder: pyramid_tensor[:, :, :, :]})
                self.session.run(initter)
                #self.session.run(self.precompile_list, feed_dict=feed_dict)
        feed_dict = dict({self.input_placeholder: pyramid_tensor[:, :, :, :]})

        result = self.session.run(self.compiled_list[3:7], feed_dict=feed_dict)

        return result

    def callback(self,
                 frame,
                 cam_id,
                 depth=2
                 ):
        z_tensor = super(LineEndFilter, self).callback(frame, cam_id)
        tensors = self.run(z_tensor)
        return [tensors[x][0] for x in range(4)]


if __name__ == '__main__':
    filter = LineEndFilter()

    filter.run_camera(cam=r"C:\\Users\\joshm\\Videos\\2019-01-18 21-49-54.mp4", fps_limit=60)
    #filter.run_camera(0, size=(800,600))
    # results = filter.run_on_pictures([r'C:\Users\joshm\OneDrive\Pictures\backgrounds'], write_results=True)
