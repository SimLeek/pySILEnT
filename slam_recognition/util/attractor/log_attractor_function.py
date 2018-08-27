import math as m


def log_attractor_function(x,
                           max_positive=1.0,
                           max_negative=0.5
                           ):
    """Uses logarithms to define how distance from the central edge effects the weights attached to each location in the
        tensor. The equation is: -log_2(x^2+((x-1)^2)/(2^p))+log_2((x-1)^2+(x^2)/(2^n), which creates this curve:
               _|
        ,,,---/ |.
        ----------\---------
                |  \_,---'''
                |
        Don't you wish you had curves like that?

        This is better than a linear equation, because you don't want far away points effecting your local edges.

        :param x: real number input. No imaginary numbers allowed, I'm trying to keep things real here.
        :param max_positive: Maximum positove number that this equation will generate.
        :param max_negative: Maximum negative number, in magnitude, that this equation will generate.
        """

    return -m.log(x ** 2 + ((x - 1) ** 2) / (2 ** (max_positive + x)), 2) + \
           m.log((x - 1) ** 2 + (x ** 2) / (2 ** (max_negative - 1 + x)), 2)
