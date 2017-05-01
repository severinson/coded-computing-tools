import unittest
import tempfile
import numpy as np
import numtools

class NumTests(unittest.TestCase):
    '''Tests of the numtools module.'''

    def test_inverse_cdf(self):
        cdf = lambda x: np.log(x)
        eps = 0.00000001
        inv_cdf = lambda x: numtools.numerical_inverse(cdf, x, 1, np.e, eps=eps, discrete=False)
        for x in np.linspace(0.0, 1.0):
            self.assertAlmostEqual(inv_cdf(x), np.exp(x))

    def test_inverse_cdf_integer(self):
        cdf = lambda x: x / 100
        inv_cdf = lambda x: numtools.numerical_inverse(cdf, x, 0, 100, discrete=True)
        for x in np.linspace(0.0, 1.0):
            self.assertEqual(inv_cdf(x), round(100 * x))
