import pandas as pd
import numpy as np

data = pd.read_csv('./c2/c2_1.csv')

data = data.to_numpy()[:,1:]

for i,(real,imag) in enumerate(data):
    
    ang = np.angle(complex(real,imag),deg=True)
    if ang<0:
        ang = 360+ang
        #ang = 2*np.pi+ang
    else:
        pass
    ang = ang-180
    #ang = ang-np.pi
    print(i,ang)
