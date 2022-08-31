
import matplotlib.pyplot as plt
import numpy as np

from image.convolution import convolve_2d
from image.image_ops import ImageOps
from image.kernel import mean_kernel, sobel_kernel_x, sobel_kernel_y


class SobelEdgeExtraction:
    """
    Primitive form of edge extraction, extracts horizontal and vertical gradient components of the image 
    and calculates their arrow vector for each pixel in the image.
    """

    def __init__(self, noise_mask_size=3, pixel_threshold=75.0) -> None:
        self.noise_mask_size = noise_mask_size
        self.pixel_threshold = pixel_threshold

    def extract(self, image_data: np.ndarray):
        """
        Extracts edge from an image.
        Parameters:
            image_data (np.ndarray): Image pixel data.
            noise_mask_size (float): X and Y size of the mask initially used to blurthe image to reduce noise.
            pixel_threshold (bool): Arbitary threshold again used in a primitive way to reduce edge noise,
                                    edge pixels below this value will be rejected. 
        Returns:
            image (np.ndarray): Edge data of shape of input image.
        """
        image_data = ImageOps.to_grayscale(image_data)

        # Denormalize grayscale if necessary
        if np.max(image_data) <= 1.0:
            image_data = (image_data * 255).astype(np.int8)

        mask = mean_kernel((self.noise_mask_size, self.noise_mask_size))
        output = convolve_2d(image_data, mask)

        horizontial_mask = sobel_kernel_x()

        vertical_mask = sobel_kernel_y()

        output_h = convolve_2d(output, horizontial_mask)

        output_v = convolve_2d(output, vertical_mask)

        edges = np.sqrt(np.square(output_h) + np.square(output_v))

        # Threshold
        edges[edges < self.pixel_threshold] = 0

        return edges
