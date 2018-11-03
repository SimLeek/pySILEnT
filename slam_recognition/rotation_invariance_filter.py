from slam_recognition.line_end_filter import LineEndFilter
import tensorflow as tf
from slam_recognition.regulator_tensor.gaussian_regulator_tensor import regulate_tensor, blur_tensor


class RotationInvarianceFilter(LineEndFilter):
    callback_depth = 6

    def __init__(self, n_dimensions=2, **argv):
        """Mimics the blob cells in the lowest layer of the V1 in the neocortex, activating pixels that have high
        color difference."""

        super(RotationInvarianceFilter, self).__init__(**argv)

    def compile(self, pyramid_tensor):
        super(RotationInvarianceFilter, self).compile(pyramid_tensor)

        with tf.name_scope('RotationInvarianceFilter Compile') and tf.device('/device:GPU:0'):
            # get indices: https://stackoverflow.com/a/39223400/782170

            hsv = tf.image.rgb_to_hsv(self.compiled_list[-1])

            zero = tf.constant(0, shape=[1,1,1],dtype=tf.float32)
            where = tf.not_equal(hsv, zero)
            indices = tf.where(where)
            values = tf.gather_nd(hsv, indices)
            sparse = tf.SparseTensor(indices, values, hsv.shape)
            #todo: sort indices

            rots = tf.map_fn(lambda i: sparse[i[0]][i[1]], indices, dtype=sparse.dtype)

            for ind in indices:
                print(ind)

            #self.compiled_list.append(compiled_line_end)


if __name__ == '__main__':
    filter = RotationInvarianceFilter()

    filter.run_camera()
