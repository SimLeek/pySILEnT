from slam_recognition.line_end_filter import LineEndFilter
import tensorflow as tf
import numpy as np
import cv2
# todo: add boosting to fix inputting full semantic while keeping sparsity!!!

if False:
    from typing import List

def dense_tensor_to_sparse(dense):
    zero = tf.constant(0, dtype=tf.float32)
    where = tf.not_equal(dense, zero)
    indices = tf.where(where)
    values = tf.gather_nd(dense, indices)

    dense_shape_list = dense.get_shape().as_list()

    return tf.SparseTensor(indices, values, dense_shape_list)

def dense_batch_to_sparse(dense):
    zero = tf.constant(-1, dtype=tf.float32)
    tensor_list = tf.unstack(dense)
    sparse_tensor_list = []
    for t in tensor_list:
        where = tf.not_equal(t, zero)
        indices = tf.where(where)
        values = tf.gather_nd(t, indices)

        dense_shape_list = t.get_shape().as_list()

        sparse_tensor_list.append(tf.SparseTensor(indices, values, dense_shape_list))

    return sparse_tensor_list

def double_sparse_tensor_list_items(
        sparse_list# type: List[tf.SparseTensor]
):
    for s in sparse_list:
        s._dense_shape = s.dense_shape*2

class WhileSparse(object):

    @staticmethod
    def is_index_within_indices(sparse_tensor, axis=0):
        def within(index, *k, **kv):
            return index < tf.shape(sparse_tensor.indices)[axis]
        return within

    @staticmethod
    def is_within_list(the_list):
        def within(index, *k, **kv):
            return index<len(the_list)
        return within

    @staticmethod
    def relative_distance_transform():
        def transform(index,  # type: int
                      sparse_tensor  # type: tf.SparseTensor
                      ):
            return_list = []
            pos = sparse_tensor.indices[index]
            val = sparse_tensor.values[index]

            #y'know, at this point, just make a new sparse tensor and return that
            #sparse_tensor._values = tf.ones([50])*255
            sparse_tensor._indices = tf.map_fn(lambda x: x,
                                               sparse_tensor.indices)
            #sparse_tensor.values[index] = val

            #sparse_tensor_list[index] = sparse_tensor
            #tens = tf.SparseTensor(indices=ind, values=sparse_tensor.values, dense_shape=sparse_tensor.dense_shape)

            return index + 1, sparse_tensor
        return transform

class SpatialInvarianceFilter(LineEndFilter):
    callback_depth = 6

    def __init__(self, n_dimensions=2, **argv):
        super(SpatialInvarianceFilter, self).__init__(**argv)
        self.tensor_return_type.append(tf.Tensor)

        self.sparse_tensor_display_program = None
        self.sparse_tensor_display_input_placeholder = None

    def compile(self, pyramid_tensor):
        super(SpatialInvarianceFilter, self).compile(pyramid_tensor)

        with tf.name_scope('SpatialInvarianceFilterCompile'):
            hsv_val = tf.maximum((self.compiled_list[-1][..., 0] + self.compiled_list[-1][..., 1] + self.compiled_list[-1][..., 2]), 0)
            hsv = tf.image.rgb_to_hsv(self.compiled_list[-1])

            # hsv_sat = hsv[..., 1] * 255 not useful
            hsv_hue = tf.where(hsv_val >0.0, (hsv[..., 0]) * 255, tf.ones_like(hsv[..., 0])*-1)

            current_sparse_list = dense_batch_to_sparse(hsv_hue)
            double_sparse_tensor_list_items(current_sparse_list)

            zcon = tf.constant(0)

            '''for s in range(len(current_sparse_list)):
                loops, new_tensors_full = tf.while_loop(WhileSparse.is_index_within_indices(current_sparse_list[s]),
                                                        WhileSparse.relative_distance_transform(),
                                                        (zcon, current_sparse_list[s]))
                current_sparse_list[s] = new_tensors_full'''

            #result = tf.sparse_tensor_to_dense(current_sparse_list)

            self.compiled_list.append(current_sparse_list)

        # SparseTensors aren't good in TensorFlow. It'd be better to just use the CPU here. I'll need to implement run

    def compile_sparse_tensor_displayer(self, sparse_tensor_batch):
        input_placeholder = [tf.sparse_placeholder(tf.int64) for _ in range(len(sparse_tensor_batch))]
        with tf.name_scope("makeImageFromSparse"):
            output = [tf.sparse_tensor_to_dense(inp) for inp in input_placeholder]
            output = tf.stack(output)
        return input_placeholder, output


    def run(self, pyramid_tensor):
        result = super(SpatialInvarianceFilter, self).run(pyramid_tensor)
        sparse_tensor_batch = result[-1] #type: List[tf.SparseTensorValue]
        for stb in range(len(sparse_tensor_batch)):
            sparse_tensor = sparse_tensor_batch[stb]
            new_indices = []
            new_values = []
            for i in range(len(sparse_tensor.indices)):
                for j in range(len(sparse_tensor.indices)):
                    new_indices.append(sparse_tensor.indices[j]-sparse_tensor.indices[i]+sparse_tensor.dense_shape/2)
                    new_values.append((sparse_tensor.values[j]))
            np_indices = np.array(new_indices)
            if np_indices.size>0:
                new_indices, indices_for_vals = np.unique(np_indices, axis=0, return_index=True)
                new_values = np.take(new_values, indices_for_vals)
            else:
                new_indices = np.ndarray((0,2))
                new_values = np.ndarray((0,))
            sparse_tensor_batch[stb] = tf.SparseTensorValue(indices=new_indices,
                                                            values=new_values,
                                                            dense_shape=sparse_tensor.dense_shape)



        if self.sparse_tensor_display_program == None or self.sparse_tensor_display_input_placeholder == None:
            self.sparse_tensor_display_input_placeholder, self.sparse_tensor_display_program = self.compile_sparse_tensor_displayer(sparse_tensor_batch)

        feed_dict = {}

        for i in range(len(self.sparse_tensor_display_input_placeholder)):
            feed_dict[self.sparse_tensor_display_input_placeholder[i]] = sparse_tensor_batch[i]

        out = self.session.run(self.sparse_tensor_display_program, feed_dict = feed_dict)
        result[-1]=out
        # cpu stuff here
        return result


if __name__ == '__main__':
    filter = SpatialInvarianceFilter()

    #filter.run_camera(cam="C:\\Users\\joshm\\Videos\\TimelineHex.mov")
    filter.run_camera(cam=0)
    #results = filter.run_on_pictures([r'C:\Users\joshm\OneDrive\Pictures\backgrounds'], write_results=True)
    #cv2.waitKey()