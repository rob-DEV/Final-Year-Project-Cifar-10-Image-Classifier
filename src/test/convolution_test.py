import unittest

import numpy as np
from image.convolution import convolve_2d
from image.kernel import sobel_kernel_x


class ConvolutionTests(unittest.TestCase):

    def test_simple_convolution(self):
        image = np.array([
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1]
        ])

        sobel = sobel_kernel_x()
        convolved_image = convolve_2d(image, sobel)
        self.assertEqual(convolved_image.max(), 0)

    def test_convolution_keep_size(self):
        image = np.array([
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1]
        ])

        sobel = sobel_kernel_x()
        convolved_image = convolve_2d(image, sobel, keep_size=True)
        self.assertEqual(convolved_image.shape, image.shape)


if __name__ == '__main__':
    unittest.main()
