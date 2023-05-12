import unittest
import numpy as np
import pandas as pd

from resummation import *

# Load the mpn energies from the scan path dataset.
data = pd.read_csv('./b2_scan.csv')
E_mpn = np.array(data[:]['1'])
delta_mpn = np.insert(E_mpn[1:] - E_mpn[:-1],0,E_mpn[0])

class TestResummation(unittest.TestCase):
    def test_linear_pade(self):
        """Compare linear pade.
        """
        resum = LinearPade.build(delta_mpn, 2, 2)
        E = resum(1.0)
        self.assertAlmostEqual(E, -49.32100056837689)

    def test_linear_borel(self):
        """Compare linear borel.
        """
        resum = LinearBorel.build(delta_mpn, 2, 2)
        E = resum(1.0)
        self.assertAlmostEqual(E, -49.3570640140236)

    def test_quadratic_pade(self):
        """Compare quadratiic pade.
        """
        resum = QuadraticPade.build(delta_mpn, 1, 1, 1)
        E = resum(1.0)

    def test_quadratic_borel(self):
        """Compare quadratic borel.
        """
        resum = QuadraticBorel.build(delta_mpn, 1, 1, 1)
        E = resum(1.0)

    def test_meijerg(self):
        """Compare MeijerG.
        """
        resum = MeijerG.build(delta_mpn, 5)
        E = resum(1.0)

if __name__ == "__main__":
    unittest.main()            
