import itertools
import numpy as np
import tensorflow as tf

are_dimensions_reversed = True

def from_shape(
        shape # type: tf.TensorShape
         ):
    global are_dimensions_reversed
    dimension_list = shape.as_list()[1:-1]
    index_tensor = np.zeros(dimension_list+[len(dimension_list)])
    dimension_ranges = [[x for x in range(d)] for d in dimension_list]
    for xyz in itertools.product(*dimension_ranges):
        if are_dimensions_reversed:
            index_tensor[xyz] = list(reversed(xyz))
        else:
            index_tensor[xyz] = xyz
    const_index_tensor = tf.constant(index_tensor, dtype=tf.int32)
    return const_index_tensor

def from_tensor(tensor):
    return from_shape(tensor.shape)