import numpy as np

from image.convolution import convolve_2d
from image.kernel import (gaussian_kernel, sobel_kernel_x,
                                       sobel_kernel_y)
from image.image_ops import ImageOps


class CannyEdgeExtraction:
    """
    This class provides a more robust form of edge extraction and uses non-maxima suppression to reduce image noise.
    """
    STRONG_EDGE_MARKER = 255
    WEAK_EDGE_MARKER = 100
    
    def __init__(self, sigma=1.0, low_threshold=0.01, high_threshold=0.09) -> None:
        self.sigma = sigma
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        pass

    def _apply_non_max_suppression(self, magnitudes, angles):
        x, y = magnitudes.shape

        # Resultant matrix to be populated
        nms_matrix = np.zeros((x, y), dtype=np.int32)

        # Convert to degrees and clamp
        # Angles less than zero
        angle = np.rad2deg(angles)
        angle[angle < 0] += 180

        for i in range(0, x-1):
            for j in range(0, y-1):
                local_max_a = 255
                local_max_b = 255

                # Angle 0
                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    local_max_a = magnitudes[i, j+1]
                    local_max_b = magnitudes[i, j-1]
                # Angle 45
                elif (22.5 <= angle[i, j] < 67.5):
                    local_max_a = magnitudes[i+1, j-1]
                    local_max_b = magnitudes[i-1, j+1]
                # Angle 90
                elif (67.5 <= angle[i, j] < 112.5):
                    local_max_a = magnitudes[i+1, j]
                    local_max_b = magnitudes[i-1, j]
                # Angle 135
                elif (112.5 <= angle[i, j] < 157.5):
                    local_max_a = magnitudes[i-1, j-1]
                    local_max_b = magnitudes[i+1, j+1]

                if (magnitudes[i, j] >= local_max_a) and (magnitudes[i, j] >= local_max_b):
                    nms_matrix[i, j] = magnitudes[i, j]
                else:
                    nms_matrix[i, j] = 0

        return nms_matrix


    def _apply_double_threshold(self, nms_image):
        x, y = nms_image.shape

        # Determine threshold
        high_threshold = np.max(nms_image) * self.high_threshold
        low_threshold = high_threshold * self.low_threshold

        strong_edge_indices = np.where(nms_image >= high_threshold)
        weak_edge_indices = np.where(
            (nms_image <= high_threshold) & (nms_image >= low_threshold))

        thresholded_matrix = np.zeros((x, y), dtype=np.int32)

        thresholded_matrix[strong_edge_indices] = self.STRONG_EDGE_MARKER
        thresholded_matrix[weak_edge_indices] = self.WEAK_EDGE_MARKER

        return thresholded_matrix


    def _has_strong_neighbour(self, img, coord):
        # Check surrounding pixels if any pixel equals the strong value return true
        # Check 3x3 surrounding pixels
        # Will throw exceptions on borders, passing for now
        for i in range(3):
            for j in range(3):
                try:
                    x, y = coord
                    if img[x - 1 + i, y - 1 + j] == self.STRONG_EDGE_MARKER:
                        return True
                except IndexError as e:
                    pass

        return False


    def _apply_hysteresis(self, thres_img):
        x, y = thres_img.shape

        # Making a copy for testing
        hysteresis_matrix = np.array(thres_img)

        # Add existing strong values
        strong_indices = np.where(thres_img == self.STRONG_EDGE_MARKER)
        hysteresis_matrix[strong_indices] = self.STRONG_EDGE_MARKER

        for i in range(x):
            for j in range(y):
                if hysteresis_matrix[i, j] == self.WEAK_EDGE_MARKER:
                    if self._has_strong_neighbour(hysteresis_matrix, (i, j)):
                        hysteresis_matrix[i, j] = self.STRONG_EDGE_MARKER
                    else:
                        hysteresis_matrix[i, j] = 0

        return hysteresis_matrix


    def extract(self, image_data: np.ndarray, return_gaussian=False):
        """
        Extracts edge from an image.
        Parameters:
            image_data (np.ndarray): Image pixel data.
            sigma (float): Standard deviation of the gaussian blur filter.
            return_gaussian (bool): If True this method with also return the gaussian kernel, purely for visualization purposes. 
        Returns:
            image (np.ndarray): Edge data of shape of input image.
            gaussian_kernel (np.ndarray): Optional return of the gaussian kernel 
        """
        image_data = ImageOps.to_grayscale(image_data)

        # Denormalize grayscale if necessary
        if np.max(image_data) <= 1.0:
            image_data = (image_data * 255).astype(np.int8)

        # Apply gaussian filter
        gaussian_filter = gaussian_kernel(sigma=self.sigma)
        convolved_gauss = convolve_2d(image_data, gaussian_filter)

        # Calculate standard each mag/angle using sobel much like HOG
        sobel_x = sobel_kernel_x()
        sobel_y = sobel_kernel_y()

        convolved_x = convolve_2d(convolved_gauss, sobel_x)
        convolved_y = convolve_2d(convolved_gauss, sobel_y)

        magnitudes = np.sqrt(np.square(convolved_x) + np.square(convolved_y))
        angles = np.arctan2(convolved_y, convolved_x)

        # Apply NMS to eliminate thicker edges
        nms_image = self._apply_non_max_suppression(magnitudes, angles)
        thresholded_image = self._apply_double_threshold(nms_image)
        hysteresis_image = self._apply_hysteresis(thresholded_image)

        if return_gaussian:
            return hysteresis_image, gaussian_filter
        else:
            return hysteresis_image
