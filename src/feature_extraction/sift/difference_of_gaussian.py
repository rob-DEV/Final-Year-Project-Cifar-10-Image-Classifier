import numpy as np
from feature_extraction.sift.octave import Octave
from feature_extraction.sift.sift_constants import SIFTConstants


class DoG:
    def __init__(self, octave: Octave) -> None:
        blurs_in_octave = octave.blurs

        self.dogs_in_octave = []
        for blur_index in range(1, SIFTConstants.BLURS_IN_OCTAVE):
            dog = np.subtract(blurs_in_octave[blur_index], blurs_in_octave[blur_index-1])
            self.dogs_in_octave.append(dog)
