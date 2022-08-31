import numpy as np

from image.image_ops import ImageOps


class Grayscale:
    def __init__(self) -> None:
        pass

    def extract(self, image_data: np.ndarray):
        """
        Converts a RGB image to grayscale.
        Parameters:
            image_data (np.ndarray): Image pixel data in the format RGB.
        Returns:
            image (np.ndarray): Single channel image of shape image_data.
        """
        return ImageOps.to_grayscale(image_data)
