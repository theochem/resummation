import numpy as np
import pandas as pd
from resummation import *



if __name__ == "__main__":
    # Load the mpn energies from the scan path dataset.
    data = pd.read_csv('../../data/mpn/cauchy_lih_scan.csv')
    E_mpn = np.array(data[:]['4.5'])
    delta_mpn = np.insert(E_mpn[1:] - E_mpn[:-1],0,E_mpn[0])

    q_pade = QuadraticPade.build(delta_mpn, *(2,2,1))
    #E = q_pade(1)[1]
    E = q_pade(complex(1))
    print(E)
