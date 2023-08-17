import numpy as np
import pandas as pd
from resummation import *



if __name__ == "__main__":
    # Load the mpn energies from the scan path dataset.
    data = pd.read_csv('../data/mpn/be2_scan.csv')
    E_mpn = np.array(data[:]['3'])

    index_list = [2,3,4,5,6,16,46]
    for index in index_list:
        print(E_mpn[index])
