import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from feature_extraction.histogram import ColorHistogram

def extract_color_histogram_demo():
    with Image.open('data/green.jpg') as image:
        image = np.asarray(image)
        histogram = ColorHistogram.extract(image)

        channel_name = {
            0: 'Red',
            1: 'Green',
            2: 'Blue',
        }

        plt.subplot(1,4, 1)
        plt.imshow(image)
        for channel_index in range(histogram.shape[0]):
            plt.subplot(1,4, channel_index + 2)
            plt.plot(np.arange(256), histogram[channel_index], color=channel_name.get(channel_index).lower())
            plt.title(channel_name.get(channel_index))

        plt.show()