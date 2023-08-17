import numpy as np
import pandas as pd
from resummation import *



if __name__ == "__main__":
    # Load the mpn energies from the scan path dataset.
    data = pd.read_csv('../data/mpn/hf_scan.csv')
    E_mpn = np.array(data[:]['3'])
    delta_mpn = np.insert(E_mpn[1:] - E_mpn[:-1],0,E_mpn[0])

    index_list = [[1,0,0],
        [1,1,0],
        [1,1,1],
        [2,1,1],
        [2,2,1],
        [5,5,5],
        [15,15,15]]
    for index in index_list:
        q_borel = QuadraticPade.build(delta_mpn, *index)
        E = q_borel(complex(1),func='plus')
        #E = q_borel(complex(1),func='minus')
        print(E)
