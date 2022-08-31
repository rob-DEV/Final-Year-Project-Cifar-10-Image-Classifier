import matplotlib.pyplot as plt
import numpy as np


from PIL import Image
from feature_extraction.edge_extraction import SobelEdgeExtraction

def extract_sobel_edges_demo():
    with Image.open('data/face.jpg') as image:
        image = np.asarray(image)
        edges = SobelEdgeExtraction()
        edges = edges.extract(image, True)
