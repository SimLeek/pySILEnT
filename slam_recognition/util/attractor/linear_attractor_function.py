# coding:utf-8

"""
This is a lot of code for a line.
"""


def linear_attractor_function_generator(max_positive=1.0, max_negative=1.0):
    """generates a simple line going through x=0 and x=1

    :param max_positive: maximum positive value, at distance zero.
    :param max_negative: maximum negative value, at distance infinity.
    :return: linear attractor function
    """

    def linear_attractor_function(x):
        """ takes in an x value and returns its position on a ʌ curve centered at 0 and going through
        +max_positive and -max_negative.

        :param x: x-value
        :return: y-value
        """
        if x >= 0:
            return max_positive - (max_negative + max_positive) * x
        else:
            return max_positive + (max_negative + max_positive) * x

    return linear_attractor_function
