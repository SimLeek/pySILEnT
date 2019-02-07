import itertools
import math as m

import numpy as np

from slam_recognition.util.attractor import euclidian_attractor_function_generator as __euclid_function_generator


def blur_tensor(n,  # type: int
                lengths=3,  # type: Union[List[int],int]
                channels_in=3,  # type: int
                channels_out=3,  # type: int
                attractor_function=__euclid_function_generator  # type: Callable[[Real], Callable[[Real], Real]]
                ):
    """Generates an n-dimensional tensor that can be convolved on any tensor to blur it.

    It should be used like:
        a = input_image
        usage_per_area = a.convolve_using(blur_tensor)
        regulator = usage_per_area/3
        regged = regulator.dot(a)

    Fun note: I made this by accident.

     :param n: number of dimensions
     :param regulation_value: value that all values in our area will add up to
     :param width: width of our oval area
     :param height: height of our oval area
     :param channels_in: number of color channels to take in.
     :param channels_out: number of color channels to output to.
     :param attractor_function:

     """
    assert n >= 1

    attractor_function = attractor_function(n, max_negative=0)

    if isinstance(lengths, int):
        gauss_dimensional_shape = [lengths for _ in range(n)]
    else:
        gauss_dimensional_shape = [lengths[i] for i in range(n)]

    gauss = np.ndarray(shape=gauss_dimensional_shape + [channels_in, channels_out])

    for tup in itertools.product(*[range(gauss_dimensional_shape[i]) for i in range(n)]):
        vector_from_center = [(tup[t] - int(gauss_dimensional_shape[t] / 2)) for t in range(len(tup))]
        euclidian_distance = m.sqrt(sum([d ** 2 for d in vector_from_center]))

        for i in itertools.product(range(channels_in), range(channels_out)):
            gauss[tup + i] = attractor_function(euclidian_distance)

    return gauss