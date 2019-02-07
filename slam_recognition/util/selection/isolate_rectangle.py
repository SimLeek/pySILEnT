import tensorflow as tf

if False:
    from typing import List


def isolate_rectangle(tensor,  # type: tf.Tensor
                      corner_1,  # type: List[int]
                      corner_2  # type: List[int]
                      ):
    shape_to_subtract = [a + (tensor.shape[b] - corner_2[b]) for a, b in zip(corner_1, range(len(corner_2)))]
    center_shape = tensor.shape - tf.constant([0] + shape_to_subtract + [0])
    shape_to_pad = [[a, tensor.shape[b] - corner_2[b]] for a, b in zip(corner_1, range(len(corner_2)))]
    center_box = tf.pad(tf.ones(center_shape), [[0, 0]] + shape_to_pad + [[0, 0]])
    output_tensor = tensor * center_box
    return output_tensor


def pad_inwards(tensor, paddings):
    center_shape = tensor.shape - tf.reduce_sum(paddings, -1)
    center_box = tf.pad(tf.ones(center_shape), paddings)
    output_tensor = center_box * tensor
    return output_tensor
