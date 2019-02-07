import tensorflow as tf
import numpy as np

if False:
    from typing import Union

def get_dimensions( tensor # type: Union[tf.Tensor, np.ndarray]
                    ):
    if isinstance(tensor, tf.Tensor):
        dimensions = len(tensor.get_shape()) - 2
    elif isinstance(tensor, np.ndarray):
        dimensions = len(tensor.shape) - 2
    else:
        raise TypeError("Input to orientation filter must either be tensor or numpy array.")

    return dimensions