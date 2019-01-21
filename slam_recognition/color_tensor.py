def color_tensor():
    """Creates a tensor where angles closer to the end vector are more positive, and angles further are more negative.

    NOTE: scaling down and detecting different lengths of lines using elongated/oriented center surrounds might be
    better.

    :param end_vector: Desired orientation to detect the most input.
    :param size: Desired width and height of tensor.
     """


    return [[[1,-0.5,-0.5],[-0.5,1,-0.5],[-0.5,-0.5,1]]]