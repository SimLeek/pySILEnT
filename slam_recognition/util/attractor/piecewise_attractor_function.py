def piecewise_attractor_function(x, max_positive=1.0, max_negative=0.5):
    if x < 0.5:
        return max_positive
    else:
        return -max_negative