from slam_recognition.line_end_filter import LineEndFilter
import tensorflow as tf
from slam_recognition.regulator_tensor.gaussian_regulator_tensor import regulate_tensor, blur_tensor
import numpy as np


def tf_unique_2d(x):
    # https://stackoverflow.com/a/51487991/782170
    x_shape = tf.shape(x)
    x1 = tf.tile(x, (1, tf.shape(x)[0]))
    x2 = tf.tile(x, (tf.shape(x)[0], 1))

    x1_2 = tf.reshape(x1, [x_shape[0] * x_shape[0], x_shape[1]])
    x2_2 = tf.reshape(x2, [x_shape[0] * x_shape[0], x_shape[1]])
    cond = tf.reduce_all(tf.equal(x1_2, x2_2), axis=1)
    cond = tf.reshape(cond, [x_shape[0], x_shape[0]])
    cond_shape = tf.shape(cond)
    cond_cast = tf.cast(cond, tf.int32)
    cond_zeros = tf.zeros(cond_shape, tf.int32)

    # CREATING RANGE TENSOR
    r = tf.range(x_shape[0])
    r = tf.add(tf.tile(r, [x_shape[0]]), 1)
    r = tf.reshape(r, [x_shape[0], x_shape[0]])

    f1 = tf.multiply(tf.ones(cond_shape, tf.int32), x_shape[0] + 1)
    f2 = tf.ones(cond_shape, tf.int32)
    cond_cast2 = tf.where(tf.equal(cond_cast, cond_zeros), f1, f2)  #

    # multiply range with new int boolean mask
    r_cond_mul = tf.multiply(r, cond_cast2)
    r_cond_mul2 = tf.reduce_min(r_cond_mul, axis=1)
    r_cond_mul3, unique_idx = tf.unique(r_cond_mul2)
    r_cond_mul4 = tf.subtract(r_cond_mul3, 1)

    # get actual values from unique indexes
    op = tf.gather(x, r_cond_mul4)

    return op


class TranslationRotationInvarianceFilter(LineEndFilter):
    callback_depth = 1

    def __init__(self, n_dimensions=2, **argv):
        """Mimics the blob cells in the lowest layer of the V1 in the neocortex, activating pixels that have high
        color difference."""

        super(TranslationRotationInvarianceFilter, self).__init__(**argv)

    def compile(self, pyramid_tensor):
        super(TranslationRotationInvarianceFilter, self).compile(pyramid_tensor)

        with tf.name_scope('RotationInvarianceFilter Compile') and tf.device('/device:CPU:0'):
            # get indices: https://stackoverflow.com/a/39223400/782170

            hsv = tf.image.rgb_to_hsv(self.compiled_list[-1])

            hsv_val = hsv[..., -1]
            hsv_hue = hsv[..., 0]
            zero = tf.constant(0, dtype=tf.float32)
            where = tf.not_equal(hsv_val, zero)
            indices = tf.where(where)
            values = tf.gather_nd(hsv_hue, indices)

            hue_shape_list = hsv_hue.get_shape().as_list()
            batch_size = hue_shape_list[0]
            new_shape_list = [batch_size] + [int(hue_shape_list[i] * 2.0) for i in range(1, len(hue_shape_list))]

            current_sparse = tf.SparseTensor(indices, values, hue_shape_list)
            new_tensors_full = tf.SparseTensor(np.empty([0, 3]), np.empty([0]), new_shape_list)

            def foldable2(j, new_indices, new_values, index1, value1):
                with tf.name_scope("foldable2"):
                    index2 = indices[j]
                    value2 = values[j]

                    def batch_not_equal():
                        return j + 1, new_indices, new_values, index1, value1

                    def batch_equal():
                        relative_distance = index2 - index1
                        relative_orientation = value2 - value1
                        center = [0] + [int(new_shape_list[i] / 2.0) for i in range(1, len(new_shape_list))]
                        relative_position = relative_distance + center

                        def allow_fun(pos):
                            fun1 = tf.greater_equal(relative_position[pos], 0)
                            # fun2 = tf.less(relative_position[pos], hsv_hue.shape[pos]*2.0)
                            # fun = tf.logical_and(fun1, fun2)
                            return fun1

                        allow_array = np.array(range(1, len(new_shape_list)))
                        allow_fun = tf.map_fn(allow_fun, allow_array, dtype=tf.bool)
                        allowable = tf.reduce_all(allow_fun)

                        def add_to_sparse():
                            ind2 = tf.concat([new_indices, [relative_position]], 0)
                            val2 = tf.concat([new_values, [relative_orientation]], 0)
                            return ind2, val2

                        def dont_add():
                            return new_indices, new_values

                        new_inds, new_vals = tf.cond(allowable, add_to_sparse, dont_add)

                        return j + 1, new_inds, new_vals, index1, value1

                    j, new_ins, new_vals, ind1, val1 = tf.cond(tf.equal(index1[0], index2[0]), batch_equal,
                                                               batch_not_equal)
                    return j, new_ins, new_vals, ind1, val1

            def foldable1(i, full_tensor):
                index = indices[i]
                value = values[i]
                zcon = tf.constant(0)
                shape_invariants = (tf.TensorShape(None),
                                    tf.TensorShape([None, 3]),
                                    tf.TensorShape([None]),
                                    index.shape,
                                    value.shape)

                def allow_while(j, i, v, p1, p2):
                    allow = j < tf.shape(indices)[0]
                    return allow

                loops, new_indices, new_values, ind1, val1 = tf.while_loop(allow_while,
                                                                           foldable2,
                                                                           (zcon, tf.Variable(np.empty([0, 3]),
                                                                                              dtype=tf.int64),
                                                                            tf.Variable(np.empty([0]),
                                                                                        dtype=tf.float32), index,
                                                                            value),
                                                                           shape_invariants=shape_invariants)
                new_tensor = tf.SparseTensor(new_indices, new_values, new_shape_list)
                concatted_indices = tf.concat([new_indices, full_tensor.indices], 0)
                prepared_indices = tf_unique_2d(concatted_indices)

                def combine_values(p):
                    # this really isn't how this should work, and another axis should be added for degrees so multiple
                    # orientations can be seen at the same point, but at this point I don't really care that much
                    add_tensors = tf.cast(tf.sparse_slice(new_tensor, p, [1, 1]).values[0], tf.float64) + \
                                  tf.sparse_slice(full_tensor, p, [1, 1]).values[0]
                    mod_tensors = tf.floormod(add_tensors, 360)
                    return mod_tensors

                prepared_values = tf.map_fn(combine_values, prepared_indices)

                added_tensor = tf.SparseTensor(prepared_indices, prepared_values, new_shape_list)
                return i + 1, added_tensor

            loops, new_tensors_full = tf.while_loop(lambda i, a: i < tf.shape(indices)[0], foldable1,
                                                    (0, new_tensors_full))

            dense = tf.sparse_tensor_to_dense(new_tensors_full)
            dense = tf.expand_dims(dense, -1)
            dense = tf.tile(dense, [1, 1, 1, 3])
            self.compiled_list.append(dense)


if __name__ == '__main__':
    filter = TranslationRotationInvarianceFilter()

    filter.run_camera()
