import numpy as np
import pandas as pd
import copy
import math


def decdeg2dms(dd):
    mult = -1 if dd < 0 else 1
    mnt,sec = divmod(abs(dd)*3600, 60)
    deg,mnt = divmod(mnt, 60)
    d,m,s =  mult*deg, mult*mnt, mult*sec
    if abs(d)>0:
        return str(int(d))+r'$\degree$'
    elif abs(d)==0 and abs(m)>0:
        return str(int(m))+r"'"
    else:
        return str(int(s))+r'"'   

# Read data
from data.c2_data import *
#err = mp20_err
#data = mp20

data = energy
err[:,0] = err[:,0]*1000
degree = []
for i,j in enumerate(err[:,1]):
    degree.append(decdeg2dms(j))


print(degree)
