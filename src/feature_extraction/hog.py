import math

import numpy as np
from image.convolution import convolve_2d
from image.image_ops import ImageOps
from image.kernel import prewitt_kernel_x, prewitt_kernel_y
from maths.maths import next_multiple
from PIL import Image
from maths.maths import EPSILON


class HOG:

    def __init__(self, block_size=8) -> None:
        self.block_size = block_size

    def _generate_block_cell_histogram(self, magnitudes: np.ndarray, directions: np.ndarray):
        hist_vals = np.zeros(9)
        for (direction, magnitude) in zip(directions.flatten(), magnitudes.flatten()):
            # select bin based on direction (0-180)
            bin_direction_index = math.floor(direction / 20.0)
            hist_vals[bin_direction_index] += magnitude

        return hist_vals

    def _reshape_if_necessary(self, magnitudes: np.ndarray, gradients: np.ndarray):
        x, y = magnitudes.shape

        reshape_needed = False
        if(x % self.block_size != 0):
            x = next_multiple(x, self.block_size)
            reshape_needed = True

        if(y % self.block_size != 0):
            y = next_multiple(y, self.block_size)
            reshape_needed = True

        if reshape_needed:
            ori_shape = magnitudes.shape
            reshaped_array = np.full((x, y), fill_value=0.0)

            reshaped_array[:ori_shape[0], :ori_shape[1]] = magnitudes
            magnitudes = np.array(reshaped_array)

            reshaped_array[:ori_shape[0], :ori_shape[1]] = gradients
            gradients = np.array(reshaped_array)

        return magnitudes, gradients

    def extract(self, image_data: np.ndarray):
        """
        Extracts a histogram of oriented gradients for each block of the image.
        Parameters:
            image_data (np.ndarray): Image pixel data.
        Returns:
            image (np.ndarray): A flattened vector of HOG. Will a length determined by the image size / block_size
        """
        img = ImageOps.to_grayscale(image_data)

        gradients_x = convolve_2d(img, prewitt_kernel_x())
        gradients_y = convolve_2d(img, prewitt_kernel_y())

        square_gradients_x = np.square(gradients_x)
        square_gradients_y = np.square(gradients_y)

        magnitudes = np.sqrt(square_gradients_x + square_gradients_y)

        # orientation offset by 90 from 0-180 degrees for bins
        gradients = np.rad2deg(
            np.arctan(gradients_y/(gradients_x + EPSILON))) + 90.0

        magnitudes, angles = self._reshape_if_necessary(magnitudes, gradients)

        # process each block
        histogram_vector = []
        for x in range(0, magnitudes.shape[0], self.block_size):
            for y in range(0, magnitudes.shape[1], self.block_size):

                block_magitudes = magnitudes[x:x + self.block_size, y:y+self.block_size]
                block_angles = angles[x:x+self.block_size, y:y+self.block_size]

                block_histograms = []

                # no splitting version
                block_histograms = self._generate_block_cell_histogram(block_magitudes, block_angles)

                # normalize block histograms
                histo_normalized = np.divide(block_histograms, np.sqrt(
                    np.sum(np.square(block_histograms))) + EPSILON)
                histogram_vector.append(histo_normalized)

        histogram_vector = np.array(histogram_vector)

        hog_vector = np.array(histogram_vector).flatten()
        return hog_vector

    @staticmethod
    def _load_hog_visual_images():
        # Helper function for visualising HOG data.
        hog_visual_images = []

        # Map i to the file name and read in each image
        for i in range(0, 160 + 1, 20):
            with Image.open("data\\res\\hog_visualisation\\{0}.png".format(i)) as bin_im:
                hog_visual_images.append(np.asarray(bin_im))

        return np.array(hog_visual_images)

    @staticmethod
    def build_hog_image(hog_vector, block_feature_length=9):
        """
        Builds a visual representation of a HOG vector.
        Parameters:
            hog_vector (np.ndarray): A previously generated hog vector.
        Returns:
            image (np.ndarray): An image of (32xblock_size) squared showing the 
            HOG magnitude and direction extracted at that sub region of an image.
        """
        # Loading angle images
        hog_visualization_images = HOG._load_hog_visual_images()

        # Calculate hog feature vector image size based on image size
        hog_vector_shape = (
            int(hog_vector.shape[0] / block_feature_length), block_feature_length)
        hog_vector = np.reshape(hog_vector, hog_vector_shape)
        block_row_count = int(np.sqrt(hog_vector_shape[0]))

        # Create the final image black canvas
        hog_block_image = Image.new(
            "RGBA", (32 * block_row_count, 32 * block_row_count), (0, 0, 0, 255))

        # Storing here to avoid loop recreation
        alpha_channel = np.array([255, 255, 255, 0])
        zero_array = np.array([255, 255, 255, 0])

        processed_count = 0

        # Foreach of the blocks in the hog vector
        for i in range(hog_vector.shape[0]):

            bin_data_array = hog_vector[i]

            # For each bin of the feature
            for j in range(9):
                mag = bin_data_array[j]

                # Set mag of the full white image to relfect the mag of the bin
                # Switching to np copy rather than file read
                bin_image = np.array(hog_visualization_images[j])

                # Zero all alpha channel data
                bin_image[np.array_equal(
                    bin_image, alpha_channel)] = zero_array

                bin_image[bin_image > 0] = bin_image[bin_image > 0] * mag

                # Calculate pos for the glyph
                hog_pos_x = int(processed_count % block_row_count) * 32
                hog_pos_y = int(processed_count / block_row_count) * 32

                # Converting back an storing
                bin_image = Image.fromarray(bin_image, "RGBA")
                hog_block_image.paste(
                    bin_image, (hog_pos_x, hog_pos_y), bin_image)

            processed_count += 1

        return np.asarray(hog_block_image)
