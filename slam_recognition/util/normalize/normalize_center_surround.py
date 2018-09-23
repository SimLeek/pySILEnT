import itertools
import numpy as np


def normalize_tensor_positive_negative(tensor,  # type: np.ndarray
                                       positive_value=1.0,
                                       negative_value=1.0,
                                       epsilon=1e-12):
    """ Normalizes a tensor so values above zero all add up to positive_value, and values below zero add up to
    -negative_value.

    :param tensor: Input tensor to normalize.
    :param positive_value: Positive parts of the tensor will sum up to this value.
    :param negative_value: Negative parts of the tensor will sum up to this value.
    :return: Normalized tensor.
    """
    sum_pos = max(sum([abs(x) if x > 0 else 0 for x in np.nditer(tensor)]), epsilon)
    sum_neg = max(sum([abs(x) if x < 0 else 0 for x in np.nditer(tensor)]), epsilon)
    for tup in itertools.product(*[range(x) for x in tensor.shape]):
        if tensor[tup] > 0:
            tensor[tup] *= positive_value / sum_pos
        if tensor[tup] < 0:
            tensor[tup] *= negative_value / sum_neg
    return tensor
