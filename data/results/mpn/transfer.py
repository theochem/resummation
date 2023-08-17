import pandas as pd
import numpy as np


data = pd.read_csv('c2_scan.csv') 
data = data.T
data = data.loc[:,20:24]
for i in [0,2,4,6,8]:
    data.insert(i, "", ['&']*16,allow_duplicates=True)


print(data)
