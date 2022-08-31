import unittest

import numpy as np
from image.image_ops import ImageOps
from PIL import Image


class ImageOpsTests(unittest.TestCase):

    def setUp(self):
        with Image.open("data/res/dev_res/face_small.jpg") as im:
            self.rgb_image = np.asarray(im)

    def test_rgb_image_to_grayscale(self):
        grayscale = ImageOps.to_grayscale(self.rgb_image)
        self.assertEqual(grayscale.ndim, 2)

    def test_rgb_image_to_grayscale_normalised(self):
        grayscale = ImageOps.to_grayscale(self.rgb_image)

        # Assert grayscale image has only one channel 
        self.assertEqual(grayscale.ndim, 2)

    def test_grayscale_image_to_grayscale_does_not_fail(self):
        grayscale = ImageOps.to_grayscale(self.rgb_image)
        grayscale = ImageOps.to_grayscale(self.rgb_image)

        self.assertEqual(grayscale.ndim, 2)


if __name__ == '__main__':
    unittest.main()
