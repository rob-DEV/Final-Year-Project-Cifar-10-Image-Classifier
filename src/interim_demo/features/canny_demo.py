import numpy as np

from PIL import Image
from feature_extraction.canny_edge_extraction import CannyEdgeExtraction

def extract_canny_edges_demo():
    with Image.open('data/face.jpg') as image:
        image = np.asarray(image)
        canny = CannyEdgeExtraction()
        canny_edges = canny.extract(image, True)
