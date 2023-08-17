import pandas as pd
import numpy as np

real = pd.read_csv('./lih_cauchy/lih_4_5/real.csv')
imag = pd.read_csv('./lih_cauchy/lih_4_5/imag.csv')

real = real.to_numpy()[:,1:]
imag = imag.to_numpy()[:,1:]



def filter(n,L):
    order = []
    reorder = []
    for i in range(L):
        for j in range(L):
            if i+j==n-2:
                order.append((i,j))
    while len(order)>0:
        i = int((len(order)-1)/2)
        reorder.append(order[i])
        del order[i]
        
    return reorder

def pick(mpn,real,imag,threshold=10e-6):
    L = real.shape[0]
    order = filter(mpn,L)
    imag_list = []
    for i,j in order:
        imag_list.append(imag[i,j])

    if min(imag_list)>threshold:
        i = imag_list.index(min(imag_list))
        m,n = order[i]
        return m,n,real[m,n],imag[m,n]

    for m,n in order:
        if imag[m,n] <= threshold:
            return m,n,real[m,n],imag[m,n]
            break

for order in [6]:
    m,n,a,b  = pick(order,real,imag,threshold=1e-6)
    print(m+1,n+1,a,b)
