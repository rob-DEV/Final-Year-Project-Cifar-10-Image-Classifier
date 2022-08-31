import unittest

import numpy as np
from maths.maths import next_multiple, vectorized_euclidean_distance


class MathsTests(unittest.TestCase):

    def test_vectorised_distance(self):
        a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        b = np.array([[4, 5, 6], [1, 2, 3], [10, 11, 12]])

        result = vectorized_euclidean_distance(a, b)

        expected = np.array([
            [5.19,  0.00,  5.19],
            [0.00,  5.19, 10.39],
            [15.58, 10.39,  5.19]
        ])

        np.testing.assert_allclose(expected, result, rtol=0.01)


    def test_vectorised_distance_identical(self):
        a = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
        b = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])

        result = vectorized_euclidean_distance(a, b)

        np.testing.assert_array_equal(np.full((3, 3), 0), result)

    def test_next_multiple(self):
        results = [next_multiple(3,5), next_multiple(10, 9), next_multiple(-4,5)]
        expected = [5, 18, 0]

        self.assertEqual(expected, results)

if __name__ == '__main__':
    unittest.main()
