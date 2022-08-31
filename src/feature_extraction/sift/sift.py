import math
import numpy as np
import matplotlib.pyplot as plt
from feature_extraction.sift.difference_of_gaussian import DoG
from feature_extraction.sift.keypoints import Keypoints
from feature_extraction.sift.octave import Octave
from feature_extraction.sift.sift_constants import SIFTConstants
from image.convolution import convolve_2d
from image.image_ops import ImageOps

from image.kernel import gaussian_kernel
from maths.maths import EPSILON, euclidean_distance


class SIFT:
    def __init__(self) -> None:
        pass


    def _generate_octaves(self, image_data : np.ndarray) -> list[Octave]:
        print("Generating scale space...")
        octaves = []

        for octave_index in range(SIFTConstants.NUMBER_OCTAVES):
            resized_image = ImageOps.resize_down(image_data, 2 ** octave_index)
            octave = Octave(resized_image)
            octaves.append(octave)

        return octaves

    def _calculate_difference_of_gaussian(self, octaves: list[Octave]) -> list[DoG]:
        print("Calculating DoG...")
        dogs = []

        for octave_index in range(SIFTConstants.NUMBER_OCTAVES):
            dog = DoG(octaves[octave_index])
            dogs.append(dog)

        return dogs

    def _generate_keypoints(self, dogs: list[DoG]):
        print("Calculating local extrema for each octave...")
        keypoints_list = []

        for octave_index in range(SIFTConstants.NUMBER_OCTAVES):
            dog = dogs[octave_index]
            keypoints = Keypoints(dog)
            keypoints_list.append(keypoints)

        return keypoints_list

    def _calculate_keypoint_orientations(self, octave_keypoints_stack, octave_dogs_stack):
        NUMBER_BINS = 36  # 0 - 360 degrees
        for octave_index in range(len(octave_keypoints_stack)):
            for keypoint_list_index in range(len(octave_keypoints_stack[octave_index])):
                corresponding_octave_dog = octave_dogs_stack[octave_index][keypoint_list_index + 1]
                # calculate the magnitude and orientation for each keypoint
                keypoint_list = octave_keypoints_stack[octave_index][keypoint_list_index]

                keypoint_histogram_bin = np.zeros(NUMBER_BINS)
                additional_keypoints = []
                for keypoint_index in range(len(keypoint_list)):
                    x = keypoint_list[keypoint_index][0]
                    y = keypoint_list[keypoint_index][1]
                    # 3x3 chunk
                    for c_x in range(x-1, x+2):
                        for c_y in range(y-1, y+2):
                            x_prev = corresponding_octave_dog[c_x-1, c_y]
                            x_next = corresponding_octave_dog[c_y+1, c_y]
                            y_prev = corresponding_octave_dog[c_x, c_y-1]
                            y_next = corresponding_octave_dog[c_x, c_y+1]

                            gradients_x = x_next - x_prev
                            gradients_y = y_next - y_prev
                            x_gradient_square = np.square(gradients_x)
                            y_gradient_square = np.square(gradients_y)
                            magnitude = np.sqrt(
                                x_gradient_square + y_gradient_square)

                            angle = np.rad2deg(np.arctan2(
                                gradients_y, gradients_x + EPSILON)) + 180.0

                            b_index = math.floor(angle / (360.0 / NUMBER_BINS))
                            keypoint_histogram_bin[b_index] += magnitude

                    # find the max value if any values are over 80% of max (excluding max) add another keypoint
                    max_index = np.argmax(keypoint_histogram_bin)
                    max_magnitude = np.max(keypoint_histogram_bin)
                    keypoint_list[keypoint_index].append(
                        (max_index + 1) * 10.0)

                    peaks_over_80_percent_of_max = np.where(
                        (keypoint_histogram_bin > .8 * max_magnitude) & (keypoint_histogram_bin < max_magnitude))

                    # add a new keypoint to the array with orientation
                    for i in range(peaks_over_80_percent_of_max[0].shape[0]):
                        additional_keypoints.append(
                            [x, y, (peaks_over_80_percent_of_max[0][i] + 1) * 10.0])
                keypoint_list.extend(additional_keypoints)

        return octave_keypoints_stack

    def _calculate_sift_descriptors(self, octave_keypoints_stack, octave_dogs_stack):

        octave_descriptor_stack = []
        for octave_index in range(len(octave_keypoints_stack)):
            octave_descriptor_list = []
            for keypoint_list_index in range(len(octave_keypoints_stack[octave_index])):
                corresponding_octave_dog = octave_dogs_stack[octave_index][keypoint_list_index + 1]
                # firstly select a 16 by 16 window (pad the dog image by 8x8 first to stop any out of bounds errors)
                PAD_WIDTH_OFFSET = 8
                padded_octave_dog = np.pad(np.array(
                    corresponding_octave_dog), pad_width=PAD_WIDTH_OFFSET, mode='constant')

                # for each keypoint select a 16x16 region around it
                keypoint_list = octave_keypoints_stack[octave_index][keypoint_list_index]
                keypoint_descriptor_vector_list = []
                for keypoint_index in range(len(keypoint_list)):
                    x = keypoint_list[keypoint_index][0]
                    y = keypoint_list[keypoint_index][1]
                    orienation = keypoint_list[keypoint_index][2]

                    # new keypoint indexes (in the the padded array)
                    padded_x = x + PAD_WIDTH_OFFSET
                    padded_y = y + PAD_WIDTH_OFFSET
                    chunk_16_x_16 = padded_octave_dog[padded_x -
                                                      8:padded_x+8, padded_y-8:padded_y+8]

                    c_w = chunk_16_x_16.shape[0]
                    c_h = chunk_16_x_16.shape[1]

                    keypoint_descriptors = []
                    # split 16x16 into 4x4 chunks to create the 128 vector 16 x 8  = 128(bins)
                    for c_x in range(0, c_w, 4):
                        for c_y in range(0, c_h, 4):
                            NUMBER_BINS = 8
                            histogram_bin = np.zeros(NUMBER_BINS)
                            
                            sub_region = chunk_16_x_16[c_x: c_x +
                                                           4, c_y: c_y + 4]

                            gradient_x_mask = np.array([[-1, 0, 1],[-1, 0, 1],[-1, 0, 1]])
                            gradient_y_mask = np.array([[1, 1, 1],[0, 0, 0],[-1, -1, -1]])

                            gradients_x = convolve_2d(sub_region, gradient_x_mask)
                            gradients_y = convolve_2d(sub_region, gradient_y_mask)

                            x_gradient_square = np.square(gradients_x)
                            y_gradient_square = np.square(gradients_y)

                            magnitudes = np.sqrt(x_gradient_square + y_gradient_square)

                            # adding 180 to shift from 0 - 360
                            angles = np.rad2deg(np.arctan2(gradients_y, gradients_x + EPSILON)) + 180.0

                            # create rotational invariance by subtract keypoint orientation
                            angles = np.subtract(angles, orienation)

                            for (dir, mag) in zip(angles.flatten(), magnitudes.flatten()):
                                # select bin based on direction (0-180)
                                b_index = math.floor(
                                    dir / (360.0 / NUMBER_BINS + 1))  # 45 per bin
                                histogram_bin[b_index] += mag
                            
                            keypoint_descriptors.extend(histogram_bin)
                    keypoint_vector = np.array(keypoint_descriptors).flatten()   
                    # normalize and threshold
                    MAX_DESCRIPTOR_VALUE = 0.2
                    norm_keypoint_vector = (keypoint_vector / np.linalg.norm(keypoint_vector)) * MAX_DESCRIPTOR_VALUE
                    keypoint_descriptor_vector_list.append(norm_keypoint_vector)  

                octave_descriptor_list.append(keypoint_descriptor_vector_list)
            octave_descriptor_stack.append(octave_descriptor_list)
        
        return octave_descriptor_stack

    def extract(self, image_data: np.ndarray):
        # gaussian blurs
        octaves = self._generate_octaves(image_data)

        # differences of gaussian
        difference_of_gaussian = self._calculate_difference_of_gaussian(octaves)

        # keypoint localization
        keypoints = self._generate_keypoints(difference_of_gaussian)


        # each octave's keypoints are in the stack extract the orientation histograms for each
        octave_keypoints_stack = self._calculate_keypoint_orientations(keypoints, difference_of_gaussian)

        # calculate the descriptor and feature vector
        octave_sift_descriptors_stack = self._calculate_sift_descriptors(octave_keypoints_stack, difference_of_gaussian)

        return octave_keypoints_stack, octave_sift_descriptors_stack


    def match(self, sds_train, sds_test):
        octave_match_stack = []
        for octave_index in range(len(sds_test)):
            # compare against all training points
            # find 2 closest neighbours and apply Lowe's ratio
            octave_match_list = []
            # 2 from the comparsision of 4 dogs
            for descriptor_list_index in range(2):
                # list of x keypoints 128 vectors
                # two in each octave
                sds_train_list = sds_train[octave_index][descriptor_list_index]
                sds_test_list = sds_test[octave_index][descriptor_list_index]

                # foreach keypoint in test calculate distances in train against test kp and find least
                dog_keypoint_matches = []
                for test_index in range(len(sds_test_list)-1):
                    test_point_dists = []
                    for train_point_index in range(len(sds_train_list)):
                        dist = euclidean_distance(sds_train_list[train_point_index], sds_test_list[test_index])
                        test_point_dists.append((train_point_index, dist))
                    
                    # sort the distances
                    if len(test_point_dists) > 0:
                        test_point_dists = sorted(test_point_dists, key=lambda tup: tup[1])
                        closest = test_point_dists[0]
                        second_closest = test_point_dists[1]

                        if closest[1] / second_closest[1] < 0.8: # lowe ratio test
                            print("Match")
                            dog_keypoint_matches.append((test_index, closest[0])) 
                octave_match_list.append(dog_keypoint_matches)

            octave_match_stack.append(octave_match_list)
        return octave_match_stack
