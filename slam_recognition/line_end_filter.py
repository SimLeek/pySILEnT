from slam_recognition.orientation_filter import OrientationFilter
import tensorflow as tf
from slam_recognition.regulator_tensor.gaussian_regulator_tensor import regulate_tensor, blur_tensor
from slam_recognition.end_tensor import rgb_2d_end_tensors
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from slam_recognition.color_tensor import color_tensor

if False:
    from typing import Optional

def generate_constant_recovery(tensor_in, recovery_amount=4):
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

def top_percentage_points(input_color, # type: tf.Tensor
                          top_percent=0.1,
                          input_gray=None # type: Optional[tf.Tensor]
                          ):
    """turns everything except the top percentage of values in a tensor to zero."""
    if input_gray is None:
        rgb_weights = [0.3333, 0.3333, 0.3333]
        input_gray = math_ops.tensordot(input_color, rgb_weights, [-1, -1])
        input_gray = array_ops.expand_dims(input_gray, -1)

    max_pooled_in_tensor_2 = tf.nn.max_pool(input_gray, (1, input_color.shape[1], input_color.shape[2], 1),
                                            strides=(1, input_color.shape[1], input_color.shape[2], 1),
                                            padding='SAME')
    min_pooled_in_tensor_2 = -1.0 * tf.nn.max_pool(-input_gray, (1, input_color.shape[1], input_color.shape[2], 1),
                                                   strides=(1, input_color.shape[1], input_color.shape[2], 1),
                                                   padding='SAME')
    top_percent_pool = (1.0 - top_percent) * max_pooled_in_tensor_2 + ( top_percent ) * min_pooled_in_tensor_2
    resized_pool = tf.image.resize_images(top_percent_pool, input_gray.shape[1:3],
                                          method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    vis_max_2 = input_color * tf.image.grayscale_to_rgb(tf.where(tf.greater_equal(input_gray, resized_pool),
                                                                         tf.ones_like(resized_pool),
                                                                         tf.zeros_like(resized_pool)))

    return vis_max_2



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

        self.relativity_tensor_shape = [4, self.output_size[1]*2,self.output_size[0]*2,3]


    def pre_compile(self, pyramid_tensor):
        self.input_placeholder = tf.placeholder(dtype=tf.float32, shape=(pyramid_tensor.shape))


        rgb_weights = [0.3333, 0.3333, 0.3333]
        gray_float = math_ops.tensordot(self.input_placeholder, rgb_weights, [-1, -1])
        gray_float = array_ops.expand_dims(gray_float, -1)
        self.energy_values = tf.Variable(tf.ones_like(gray_float)*self.excitation_max , dtype=tf.float32)

        self.precompile_list= [self.energy_values]

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

            center_shape = compiled_line_end_0.shape - tf.constant([0,4,4,0])
            center_box = tf.pad(tf.ones(center_shape), [[0,0],[2,2],[2,2],[0,0]])
            compiled_line_end_0 = tf.where(tf.equal(center_box,1), compiled_line_end_0, tf.zeros_like(compiled_line_end_0))

            # todo: move to its own filter class
            rgb_weights = [0.3333, 0.3333, 0.3333]
            gray_float = math_ops.tensordot(compiled_line_end_0, rgb_weights, [-1, -1])
            gray_float = array_ops.expand_dims(gray_float, -1)

            # todo: replace with faster and more stable grouping areas

            memory_values = gray_float**self.energy_values.value()
            max_pooled_in_tensor = tf.nn.max_pool(memory_values, (1,3, 3,1), strides=(1,1,1,1), padding='SAME')

            has_fired = tf.where(tf.logical_and(tf.greater_equal(max_pooled_in_tensor,100) ,tf.equal(memory_values, max_pooled_in_tensor)),
                                                             tf.ones_like(max_pooled_in_tensor),
                                                             tf.zeros_like(max_pooled_in_tensor))

            fire_strength = has_fired * gray_float


            exhaustion_max = 8

            exhaustion = has_fired*255.0-fire_strength

            recovery = generate_recovery(fire_strength, self.input_based_recovery, self.constant_recovery)

            update_energy = self.energy_values.assign(
                tf.clip_by_value((self.energy_values*255-exhaustion+recovery)/255, -exhaustion_max, self.excitation_max)
            )

            has_fired2 = tf.image.grayscale_to_rgb(has_fired) * compiled_line_end_0

            top_percent_points = top_percentage_points(has_fired2, self.top_percent_pool, gray_float)

            top_idxs = tf.cast(tf.where(tf.not_equal(top_percent_points, 0)), tf.int32)

            #def body(i, relativity, slices):
            w = int(self.relativity_tensor_shape[1] / 2.0)
            h = int(self.relativity_tensor_shape[2] / 2.0)
            self.padded_firing = tf.pad(has_fired2, [[0, 0], [w, w], [h, h], [0, 0]], "CONSTANT")

            def get_slices(slice):
                if self.rotation_invariance:
                    raise NotImplementedError("This will require another variable")
                    angle_rgb = self.padded_firing[slice[0]:slice[0]+1,slice[1]:slice[1]+1,slice[2]:slice[2]+1,:]
                    angle_hue = tf.image.rgb_to_hsv(angle_rgb)[0:1,0:1,0:1,0:1]
                    angle = tf.squeeze(angle_hue)
                    angle.set_shape([1])
                    pad_rot = tf.contrib.image.transform(
                        self.padded_firing,
                        tf.contrib.image.angles_to_projective_transforms(
                            angle, tf.cast(tf.shape(self.padded_firing)[1], tf.float32), tf.cast(tf
                                                                                             .shape(self.padded_firing)[2],
                                                                                             tf.float32)
                        ))
                else:
                    pad_rot = self.padded_firing
                value_slice = pad_rot[ tf.newaxis,
                              slice[0]:slice[0]+1,
                              slice[1]:slice[1]+w*2,
                              slice[2]:slice[2]+h*2,
                              :
                          ]
                return tf.pad(value_slice, [[0,0],[slice[0], tf.cast(tf.shape(pad_rot), tf.int32)[0]-slice[0]], [0,0], [0,0], [0, 0]], "CONSTANT")
            batch_items = tf.map_fn(fn=get_slices,
                                    elems=top_idxs,
                                    dtype=tf.float32)
            added_relative = tf.squeeze(tf.reduce_sum(batch_items, 0),[0])


            relativity_tensor = tf.cast(added_relative, tf.float32)
            #    return i, tf.clip_by_value(added_relative,0,255), slices

            #i_start = tf.constant(0, dtype=tf.int64)
            #i, relativity_tensor, idxs = tf.while_loop(body, while_condition, [i_start, relativity_tensor, top_idxs],
            #                                           shape_invariants=[i_start.get_shape(),
            #                                                             tf.TensorShape(self.relativity_tensor_shape),
            #                                                             tf.TensorShape([None,None])]
            #                                           )

            self.compiled_list.extend([tf.image.grayscale_to_rgb(update_energy*(255.0/16)+127.5),has_fired2,top_percent_points,relativity_tensor, compiled_line_end_0])
            #self.compiled_list.extend([tf.clip_by_value(compiled_line_end_0, 0, 255)])

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
                #self.session.run(self.precompile_list, feed_dict=feed_dict)

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
                #self.session.run(self.precompile_list, feed_dict=feed_dict)
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
        return [frame]+[[tensors[x][y] for y in range(len(tensors[x]))] for x in range(5)]


if __name__ == '__main__':
    filter = LineEndFilter()

    #filter.run_camera(cam=r"C:\\Users\\joshm\\Videos\\2019-01-18 21-49-54.mp4")
    #filter.run_on_pictures(r"C:\\Users\\joshm\\OneDrive\\Pictures\\robots\\repr\\phone.png", resize=(-1,480))
    filter.run_camera(0, size=(800,600))
    # results = filter.run_on_pictures([r'C:\Users\joshm\OneDrive\Pictures\backgrounds'], write_results=True)
