import numpy as np

if False:
    from typing import List


def zoom_tensor_to_image_list(zoom  # type: np.ndarray
                              ):  # type: (...)->List[np.ndarray]
    """Converts an image pyramid tensor to a list of images. Can be used by cvpubsub library to display.

    :param zoom: any one or three channel tensor
    :return: a list of images.
    """
    return [np.squeeze(zoom[[slice(p, p + 1)] + [slice(None)] * (zoom.ndim - 1)]).astype(dtype=np.uint8) for p
            in range(zoom.shape[0])]
