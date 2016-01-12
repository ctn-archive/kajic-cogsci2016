# run with: python -m unittest discover

import unittest
import numpy as np
import utils


class TestSVD(unittest.TestCase):
    def setUp(self):
        self.nr_dim = 100
        self.red_dim = 10

        self.matrix = np.random.rand(self.nr_dim, self.nr_dim)
        self.red_mat = utils.reduced_matrix(self.matrix, nr_dim=self.red_dim)

    def testDimensions(self):
        computed = self.red_mat.shape
        expected = (self.nr_dim, self.red_dim)

        np.testing.assert_equal(computed, expected)

    def testNormalized(self):
        # TODO: does not test for zeros, which is a norm of zero vector
        norms = np.linalg.norm(self.red_mat, axis=1)

        np.testing.assert_allclose(norms, 1)
