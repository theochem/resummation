import numpy as np
import pandas as pd

# Read data
from data.lih_data import *
print(pd.DataFrame(mp20))
print(pd.DataFrame(mp20).to_csv('./test.csv'))

