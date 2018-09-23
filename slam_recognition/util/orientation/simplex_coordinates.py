import numpy as np


def simplex_coordinates(n):
    """ Generates the coordinates of the smallest possible shape in n dimensions. Useful for representing space with
    only positive numbers, which is helpful with networks that form links to neurons with positive values.

    From: https: // math.stackexchange.com / a / 2534608 / 324663
    From : https://people.sc.fsu.edu/~jburkardt/py_src/simplex_coordinates/simplex_coordinates1.py
    Licensing: This function is distributed under the GNU LGPL license.

    :param n: the number of dimensions.
    :return: a numpy array of length n+1 containing all the coordinates
    """
    coordinates = np.zeros([n + 1, n])

    for dimension in range(0, n):
        s = sum([c ** 2 for c in coordinates[dimension, :dimension]])
        coordinates[dimension, dimension] = np.sqrt(1.0 - s)
        for j in range(dimension + 1, n + 1):
            s = sum([c1 * c1 for c1, c2 in zip(coordinates[dimension, :dimension], coordinates[j, :dimension])])
            coordinates[j, dimension] = (-1.0 / float(n) - s) / coordinates[dimension, dimension]

    return coordinates


def axis_coordinates(n):
    return np.eye(n, n)


def above_axis_simplex_coordinates(n, axis=0):
    s = simplex_coordinates(n)
    s[:, axis] = np.absolute(s[:, axis])
    return s
