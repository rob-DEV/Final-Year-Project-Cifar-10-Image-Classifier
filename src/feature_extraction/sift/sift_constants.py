import numpy as np


class SIFTConstants:
    NUMBER_OCTAVES = 4
    BLURS_IN_OCTAVE = 5
    SIGMA = 1.6 # D. Lowe Paper
    K = np.sqrt(2)
    
    KEYPOINT_MARKER = 1
    FLAT_THRESHOLD = 0.03  # D. Lowe Paper
    HESSIAN_EDGE_THRESHOLD = 10.0  # D. Lowe Paper
