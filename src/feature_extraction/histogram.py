import numpy as np
from matplotlib import pyplot as plt


class ColorHistogram:
    def __init__(self) -> None:
        pass

    def extract(self, image_data: np.ndarray):
        """
        Extracts a RGB lookup table counting the 256 levels for each image channel.
        Parameters:
            image_data (np.ndarray): Image pixel data.
        Returns:
            histogram (np.ndarray): A 3 channel table of (3x256) populated by the count of pixel values
            at each level.
        """
        r_lut = np.array([0] * 256, dtype=np.int32)
        g_lut = np.array([0] * 256, dtype=np.int32)
        b_lut = np.array([0] * 256, dtype=np.int32)

        if image_data.ndim == 3 and image_data.shape[2] == 3:
            if np.max(image_data) <= 1.0:
                image_data = (image_data * 255).astype(np.int8)


            r_channel = image_data[:, :, 0].flatten()
            g_channel = image_data[:, :, 1].flatten()
            b_channel = image_data[:, :, 2].flatten()

            # Populate histograms
            for r, g, b in zip(r_channel, g_channel, b_channel):
                r_lut[r] += 1
                g_lut[g] += 1
                b_lut[b] += 1
        else:
            print("Color Histogram: Warning image is not of RGB format!")


        return np.array([r_lut, g_lut, b_lut])
