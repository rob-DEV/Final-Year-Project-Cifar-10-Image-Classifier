import unittest

import numpy as np
from PIL import Image

from feature_extraction.histogram import ColorHistogram
import matplotlib.pyplot as plt

class CifarDatasetTests(unittest.TestCase):

    def test_color_histogram(self):
        with Image.open("data/res/dev_res/face_small.jpg") as im:
            rgb_image = np.asarray(im)
            histo = ColorHistogram()
            rgb_histogram_array = histo.extract(rgb_image)

        plt.plot(np.arange(256), rgb_histogram_array[0], 'r')
        plt.plot(np.arange(256), rgb_histogram_array[1], 'g')
        plt.plot(np.arange(256), rgb_histogram_array[2], 'b')

        # Assert pixel sums equal the pixels in the image
        sum_of_histogram_values = rgb_histogram_array.sum()
        self.assertEqual(rgb_image.size, sum_of_histogram_values)
        

if __name__ == '__main__':
    unittest.main()
