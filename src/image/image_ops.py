import numpy as np
import matplotlib.pyplot as plt


class ImageOps:
    """
    Provides a series of static utility methods for simple image operations on the CIFAR-10 dataset
    including conversion to grayscale and resizing
    """
    
    @staticmethod
    def to_grayscale(image_data: np.ndarray) -> np.ndarray:
        """
        Converts a RGB image to grayscale.
        Parameters:
            image_data (np.ndarray): Image pixel data in the format RGB.
        Returns:
            image (np.ndarray): Single channel image of shape image_data.
        """

        # Check the amount of channels in the images and only perform the
        # conversion if the image has 3 channels (RGB)
        if image_data.ndim == 3 and image_data.shape[2] == 3:
            rgb_weights = [0.2989, 0.5870, 0.1140]
            image_data = np.dot(image_data[..., :3], rgb_weights)
        else:
            print("Grayscale: Warning image is not of RGB format!")
        return image_data


    @staticmethod
    def resize_down(image_data: np.ndarray, factor: int = 2) -> np.ndarray:
        """
        Decreases the size of an image using indexing without interpolation by a scale factor.
        Parameters:
            image_data (np.ndarray): Image pixel data.
            factor (int): Factor by which image is reduced
        Returns:
            image (np.ndarray): Reduced image pixel data
        """
        factor = int(factor)
        return image_data[::factor, ::factor]
