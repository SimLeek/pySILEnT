# coding:utf-8

"""
This competitive attractor function matches forces in euclidian space as closely as possible.
"""


def euclidian_attractor_function_generator(n, max_positive=1.0, max_negative=1.0):
    """generates a physics based competitive attractor function for n-dimensional space.

    :param n: number of dimensions.
    :param max_positive: maximum positive value, at distance zero.
    :param max_negative: maximum negative value, at distance infinity.
    :return: euclidian attractor function for n dimensions
    """

    def n_dimensional_euclid_function(x):
        """ takes in distance and returns competition value between local and global forces.
        In the case of negative distances, repulsive forces are used.

        :param x: distance value
        :return: local and global force competition based on distance
        """
        if x >= 0:
            return (max_positive + max_negative) / (((2 * x ** (n - 1)) + 1) ** (n - 1)) - max_negative
        else:
            x = -x
            return -(max_positive + max_negative) / (((2 * x ** (n - 1)) + 1) ** (n - 1)) + max_negative

    return n_dimensional_euclid_function
