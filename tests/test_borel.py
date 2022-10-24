import unittest
import numpy as np

from resummation.borel import BorelPade

Emp0 = -23.181724 
mpn = np.array([-25.109634, 
    -0.134539, 
    -0.021331, 
    -0.019466, 
    -0.005941, 
    -0.005162, 
    -0.003411, 
    -0.002142, 
    -0.001477, 
    -0.000844])

class TestPade(unittest.TestCase):
    def test_pade(self):
        """Compare palin pade with robust pade.
        """
        bp = BorelPade.build(mpn, 2, 2)
        E = bp(1.0)
        E = bp(1.0)[0]+Emp0
        print(E)
        #np.testing.assert_allclose(Ep, Er, rtol=1e-6)

if __name__ == "__main__":
    unittest.main()            