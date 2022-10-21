import unittest
import numpy as np

from resummation.meijerg import MeijerG

class TestPade(unittest.TestCase):
    def test_mpmath_pade(self):
        ### compare with mpmath pade
        mpn = np.array([-78.304449, -0.344056, -0.008168, -0.037373])
        for order in [1]:
            meijerg = MeijerG.build(mpn, order)
            print(order,meijerg(1)-20.145516)

            #np.testing.assert_allclose(a1, a2, rtol=1e-14)

if __name__ == "__main__":
    unittest.main()            