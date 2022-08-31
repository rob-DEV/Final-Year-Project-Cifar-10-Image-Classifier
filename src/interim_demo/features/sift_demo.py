import numpy as np

from PIL import Image, ImageOps
from feature_extraction.sift.sift import SIFT

def extract_sift_demo():
    with Image.open('data/face_small.jpg') as image:
        image = ImageOps.grayscale(image)
        image = np.asarray(image)
        sift = SIFT()
        keypoints, descriptors = sift.extract(image)
