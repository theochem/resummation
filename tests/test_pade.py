import unittest
import numpy as np

from resummation.pade import Pade

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
        p_pade = Pade.plain(mpn, 1, 1)
        r_pade = Pade.robust(mpn, 1, 1)
        Ep = p_pade(1)+Emp0
        Er = r_pade(1)+Emp0
        print(Ep,Er)
        np.testing.assert_allclose(Ep, Er, rtol=1e-6)

    def test_pade_table(self):
        p_table = np.zeros((4,4))
        r_table = np.zeros((4,4))
        for m in range(4)[0:]:
            for n in range(4)[1:]:
                p_pade = Pade.plain(mpn, m, n)
                r_pade = Pade.robust(mpn, m, n)
                p_table[m,n] = p_pade(1)+Emp0
                r_table[m,n] = r_pade(1)+Emp0
        print(p_table)
        print(r_table)



if __name__ == "__main__":
    unittest.main()            