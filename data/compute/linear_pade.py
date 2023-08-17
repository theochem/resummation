import numpy as np
import pandas as pd
from resummation import *



if __name__ == "__main__":
    # Load the mpn energies from the scan path dataset.
    data = pd.read_csv('../data/mpn/cauchy_lih_scan.csv')
    E_mpn = np.array(data[:]['0.9'])
    delta_mpn = np.insert(E_mpn[1:] - E_mpn[:-1],0,E_mpn[0])
    index_list = [[1,1],
        [2,1],
        [2,2],
        [3,2],
        [3,3],
        [8,8],
        [23,23]]
    for index in index_list:
        l_pade = LinearPade.build(delta_mpn, *index)
        E = l_pade(1)
        print(E)
