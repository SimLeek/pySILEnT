from tensorflow.python.ops import array_ops
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
import tensorflow as tf

def to_channels(images, num_channels=3, name=None):
    # remade from tensorflow core
    with ops.name_scope(name, 'grayscale_to_channels', [images]) as name:
        images = ops.convert_to_tensor(images, name='images')
        rank_1 = array_ops.expand_dims(array_ops.rank(images) - 1, 0)
        shape_list = ([array_ops.ones(rank_1, dtype=dtypes.int32)] +
                      [array_ops.expand_dims(num_channels, 0)])
        multiples = array_ops.concat(shape_list, 0)
        rgb = array_ops.tile(images, multiples, name=name)
        rgb.set_shape(images.get_shape()[:-1].concatenate([num_channels]))
        return rgb
