import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import math

# Initialize the matplotlib figure
sns.set_theme(style="whitegrid")
cmap = ['tab:red', 'tab:blue']

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
from data.hf_data import *
err = mp6_err
data = mp6
err[:,0] = err[:,0]*1000
degree = []
for i,j in enumerate(err[:,1]):
    degree.append(decdeg2dms(j))
for i in range(5):
    data[i,:] = (data[i,:] - fci[i])*1000

# Plot
dim = len(data[0])
w = 0.75
dimw = w / dim
method = ['MPn',r'Pad$\'e$',r'Borel-Pad$\'e$','Meijer-G']
fig, ax = plt.subplots()
x = np.arange(len(data))
for i in range(len(data[0])):
    y = [d[i] for d in data]
    if i==2:
        ax.bar(x + i * dimw, y, dimw, bottom=0, yerr=err[:,i-2],label=method[i])
    elif i==3:
        p = ax.bar(x + i * dimw, y, dimw, bottom=0, label=method[i])
        ax.bar_label(p, labels=degree, padding=1,fontsize=16)
    else:
        ax.bar(x + i * dimw, y, dimw, bottom=0, label=method[i])

ax.set_xticks(x+dimw*1.5)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

x_label = [r'$1$',r'$1.5$',r'$2.0$',r'$2.6$',r'$3.0$']
ax.set_xticklabels(x_label)

ax.set_xlabel(r'Bond Length ($r_e$)',fontsize=16)
ax.set_ylabel('Energy Correlation [mhartree]',fontsize=16)

#ax.set_ylim([-10,40])
ax.legend(fontsize=12,bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=4, mode="expand", borderaxespad=0.)
plt.savefig('./figure.png', dpi=500, bbox_inches='tight',pad_inches=0)
