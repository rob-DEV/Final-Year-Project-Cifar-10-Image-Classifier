import matplotlib.pyplot as plt
import numpy as np


from PIL import Image
from feature_extraction.hog import HOG

def extract_hog_vector_demo():
    with Image.open('data/face.jpg') as image:
        image = np.asarray(image)
        hog = HOG()
        hog_vector = hog.extract(image)

        hog_image = hog.build_hog_image(hog_vector, image.shape)

        plt.subplot(1,2,1)
        plt.imshow(image)
        plt.subplot(1,2,2)
        plt.imshow(hog_image)

        plt.show()
