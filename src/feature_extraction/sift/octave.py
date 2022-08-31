import numpy as np
from feature_extraction.sift.sift_constants import SIFTConstants
from image.convolution import convolve_2d
from image.kernel import gaussian_kernel


class Octave:    
    def __init__(self, image) -> None:
        # Generate blurs in octave
        self.blurs = []
        for blur_index in range(SIFTConstants.BLURS_IN_OCTAVE):
            kernel_sigma_j = gaussian_kernel(SIFTConstants.SIGMA + (SIFTConstants.K * (blur_index + 1)))
            blurred = convolve_2d(image, kernel_sigma_j, keep_size=True)
            self.blurs.append(blurred)