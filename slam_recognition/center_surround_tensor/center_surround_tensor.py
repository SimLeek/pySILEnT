import itertools
import math as m

import numpy as np


def center_surround_tensor(ndim,  # type: int
                           center_in,  # type: List[int]
                           center_out,  # type: List[int]
                           surround_in,  # type: List[int]
                           surround_out  # type: List[int]
                           ):
    """Generates a multi-channel center surround center_surround matrix. Useful for isolating or enhancing edges.

    Note: center-surround tensors with 11 or more dimensions may take a while to generate. Make sure you cache those.

    :param center_in: input tensor of ints. shape of tensor depends on desired input/output.
    :param center_out:
    :param surround_in:
    :param surround_out:
    """
    assert ndim >= 1

    center_surround = np.ndarray(shape=[3 for _ in range(ndim)] + [len(center_in), len(center_out)])

    total = 0
    for tup in itertools.product(*[range(3) for _ in range(ndim)]):
        inv_manhattan_dist = sum([abs(t - 1) for t in tup])
        if inv_manhattan_dist == 0:
            center_surround[tup] = [[0 for _ in center_out] for _ in center_in]
        else:
            euclidian_dist = 1. / m.sqrt(inv_manhattan_dist)
            center_surround[tup] = [[o * i * euclidian_dist for o in surround_out] for i in surround_in]
            total += euclidian_dist
    center_index = tuple([1 for _ in range(ndim)])
    center_surround[center_index] = [[o * i * total for o in center_out] for i in center_in]
    return center_surround