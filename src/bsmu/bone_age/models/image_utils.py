import numpy as np


def normalized_image(image):
    """
    :param image: two- or three-dimensional image
    :return: normalized to [0, 1] image
    """
    image_min = image.min()
    if image_min != 0:
        image = image - image_min
    image_max = image.max()
    if image_max != 0 and image_max != 1:
        image = image / image_max
    return image


def normalized_8U_image(image):
    """
    :param image: two- or three-dimensional image
    :return: normalized to [0, 255] image
    """
    return (normalized_image(image) * 255).astype(np.uint8)


def normalized_8UC3_image(image):
    """
    :param image: two-dimensional image 
    :return: normalized to [0, 255] three-dimensional image
    """
    assert len(image.shape) == 2, 'two-dimensional images are only supported'

    image = normalized_8U_image(image)
    return np.stack((image,) * 3, axis=-1)
