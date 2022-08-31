import numpy as np

from feature_extraction.sift.difference_of_gaussian import DoG
from feature_extraction.sift.sift_constants import SIFTConstants


class Keypoints:

    def __init__(self, dog: DoG) -> None:

        # Calculate local minima maxima for each 3x3 dog chunk
        self.keypoints = []
        for dog_index in range(1, len(dog.dogs_in_octave) - 1):
            left_dog = dog.dogs_in_octave[dog_index - 1]
            center_dog = dog.dogs_in_octave[dog_index]
            right_dog = dog.dogs_in_octave[dog_index + 1]

            dog_keypoints = []
            for x in range(0, center_dog.shape[0] - 3, 1):
                    for y in range(0, center_dog.shape[1] - 3, 1):
                        if self._is_local_exterma(x,y,left_dog, center_dog, right_dog):
                            print("Is Local Extrema")
                            extrema, hessian_matrix = self._calculate_image_derivatives(x, y, center_dog)

                            # Calculate taylor derived offset
                            try:
                                offset = -np.linalg.inv(hessian_matrix).dot(extrema)
                                contrast = np.sum(
                                    center_dog[x, y] + .5 * extrema.dot(offset))

                                # edge thresholding
                                a, b = np.linalg.eig(hessian_matrix)
                                r = a[1] / a[0]
                                ratio = np.square(r+1) / r

                                if abs(contrast) >= SIFTConstants.FLAT_THRESHOLD and ratio > SIFTConstants.HESSIAN_EDGE_THRESHOLD:
                                    if x + offset[0] < center_dog.shape[0] and y + offset[1] < center_dog.shape[1]:
                                        dog_keypoints.append([x, y])
                            except Exception as a:
                                pass
            self.keypoints.append(dog_keypoints)

            import matplotlib.pyplot as plt
            plt.imshow(center_dog, cmap='gray')
            x = []
            y = []

            for index in range(len(dog_keypoints)):
                kx = dog_keypoints[index][0]
                ky = dog_keypoints[index][1]

                x.append(kx)
                y.append(ky)


            for p,q in zip(x,y):
                x_cord = q
                y_cord = p
                plt.scatter([x_cord], [y_cord])
            
            plt.show()

    
    def _is_local_exterma(self, x, y, left_dog, dog, right_dog):
        pixel_to_compare = dog[x, y]

        left_chunk = left_dog[x: x + 3, y: y + 3]
        target_chunk = dog[x: x + 3, y: y + 3]
        right_chunk = right_dog[x: x + 3, y: y + 3]

        flattened_neigbours = np.array(
            [left_chunk, target_chunk, right_chunk]).flatten()

        if np.max(flattened_neigbours) == pixel_to_compare or np.min(flattened_neigbours) == pixel_to_compare:
            return True

        return False

    def _calculate_image_derivatives(self, x, y, dog):
        # D(x) = D + dD^T/dx * x + 1/2*x^T * d^2(D) / d(x^2) * x
        # as per the paper D Lowe.
        # D (exterma) = D + 1/2 * pd(D^T) / pd(x)
        current = dog[x, y]
        x_prev = dog[x-1, y]
        x_next = dog[x+1, y]
        y_prev = dog[x, y-1]
        y_next = dog[x, y+1]

        # first taylor derivative
        dx = 0.5 * (x_next - x_prev)
        dy = 0.5 * (y_next - y_prev)

        # second derivative using lapacian masking [1, -2, 1]
        # next - (2 x current) + previous
        dxx = x_next - (2.0 * current) + x_prev
        dyy = y_next - (2.0 * current) + y_prev

        dxy = ((dog[x+1, y+1] - dog[x+1, y-1]) -
               (dog[x-1, y+1] - dog[x-1, y-1])) * .25

        extrema = np.array([dx, dy])
        hessian_matrix = np.array([
            [dxx, dxy],
            [dxy, dyy]
        ])

        return extrema, hessian_matrix
