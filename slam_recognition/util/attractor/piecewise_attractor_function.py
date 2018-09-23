def piecewise_attractor_function(x, max_positive=1.0, max_negative=0.5):
    """ This shunts the input to one value or the other depending on if its greater or less than 0.5.

    :param x: x-value
    :return: y-value
    """
    if x < 0.5:
        return max_positive
    else:
        return -max_negative
