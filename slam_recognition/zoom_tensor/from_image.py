import math as m

import numpy as np
from scipy import ndimage

if False:
    from typing import List


def image_to_zoom_tensor(image,  # type: np.ndarray
                         num_colors,  # type: int
                         center_dimensions,  # type: List[int]
                         scale  # type: float
                         ):  # type: (...)->np.ndarray
    """Converts an image to image pyramid.

    Each image in the image pyramid will have the same number of pixels. They will take larger and larger sections
     of the input image and scale them down to the center_dimensions size.

    Example:

    _____________________________________________________________________________________
    |                    |                    |                    |                    |
    |     **********     |        ****        |                    |                    |
    |   *            *   |      *      *      |        ****        |                    |
    | *                * |   *            *   |      *      *      |         **         |
    | *                * |   *            *   |      *      *      |         **         |
    |   *            *   |      *      *      |        ****        |                    |
    |     **********     |        ****        |                    |                    |
    |____________________|____________________|____________________|____________________|

    :param image: The input image.
    :param num_colors: The number of colors in the input image.
    :param center_dimensions: The size of the images used in the image pyramid.
    :param scale: The consecutive scales of the regions covered by the images in the image pyramid.
    :return: A tensor containing the image pyramid. Format: Pyramid[num_images, center_width, center_height, colors]
    """
    assert scale > 1, "Scale must be greater than one."
    assert num_colors > 0, "Number of colors must be greater than zero."
    for d in center_dimensions:
        assert d > 0, "Each dimension must be larger than zero."

    image_dimensions = image.shape[:-1]  # take out colors from n-dimensional image
    center_dimensions = list(reversed(center_dimensions))  # because OpenCV likes doing things backwards
    num_scales = int(
        m.ceil(max([m.log(x_i / x_c, scale) for x_i, x_c in zip(image_dimensions, center_dimensions)])))
    pyramid_tensor = np.empty([num_scales] + center_dimensions + [num_colors])
    for s in range(num_scales):
        slice_dimensions = [c * (scale ** s) for c in center_dimensions]
        image_slicers = [slice(int(max((i - c) / 2, 0)), int((i + c) / 2)) for i, c in
                         zip(image_dimensions, slice_dimensions)]
        image_slice = image[image_slicers]
        scaled_image_slice = np.empty(center_dimensions + [num_colors])
        for c in range(num_colors):
            scaled_image_slice[[slice(None)] * len(center_dimensions) + [slice(c, c + 1)]] = \
                ndimage.zoom(np.squeeze(image_slice[[slice(None)] * len(image_dimensions) + [slice(c, c + 1)]]),
                             1.0 / (scale ** s), prefilter=False, order=5)[
                    [slice(None)] * len(center_dimensions) + [np.newaxis]]
        pyramid_tensor_slicers = [slice(None) for _ in image_dimensions] + [slice(None)]
        pyramid_tensor[[slice(s, s + 1)] + pyramid_tensor_slicers] = scaled_image_slice[[np.newaxis] +
                                                                                        pyramid_tensor_slicers]
    return pyramid_tensor
