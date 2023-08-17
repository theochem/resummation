import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt


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
from data.be2_data import *
err = mp3_err
data = mp3
err[:,0] = err[:,0]*1000
degree = []
for i,j in enumerate(err[:,1]):
    degree.append(decdeg2dms(j))
for i in range(5):
    data[i,:] = (data[i,:] - fci[i])*1000


dim = len(data[0])
w = 0.75
dimw = w / dim
method = ['MPn',r'Pad$\'e$',r'Borel-Pad$\'e$','Meijer-G']


# 创建两个绘图坐标轴；调整两个轴之间的距离，即轴断点距离
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
fig.subplots_adjust(hspace=0.1)  # adjust space between axes

# 将用相同的绘图数据，在两个轴上绘制折线图
x = np.arange(len(data))
for i in range(len(data[0])):
    y = [d[i] for d in data]
    # 将用相同的绘图数据，在两个轴上绘制折线图
    if i==2:
        ax1.bar(x + i * dimw, y, dimw, bottom=0, yerr=err[:,i-2],label=method[i])
        ax2.bar(x + i * dimw, y, dimw, bottom=0, yerr=err[:,i-2],label=method[i])
    elif i==3:
        p1 = ax1.bar(x + i * dimw, y, dimw, bottom=0, label=method[i])
        p2 = ax2.bar(x + i * dimw, y, dimw, bottom=0, label=method[i])
        ax1.bar_label(p1, labels=degree, padding=1,fontsize=16)
        ax2.bar_label(p2, labels=degree, padding=1,fontsize=16)
    else:
        ax1.bar(x + i * dimw, y, dimw, bottom=0, label=method[i])
        ax2.bar(x + i * dimw, y, dimw, bottom=0, label=method[i])

# 调整两个y轴的显示范围
ax1.set_ylim(300, 330)  # outliers only
ax2.set_ylim(0, 20)  # most of the data


# 创建轴断刻度线，d用于调节其偏转角度
d = 0.5  # proportion of vertical to horizontal extent of the slanted line
kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
              linestyle="none", color='k', mec='k', mew=1, clip_on=False)
ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)


x_label = [r'$1$',r'$1.5$',r'$2.0$',r'$2.6$',r'$3.0$']
ax2.set_xticks(x+dimw*1.5)
ax2.set_xticklabels(x_label)

# Change the fontsize
for ax in [ax1,ax2]:  # example for xaxis
    for tick in ax.yaxis.get_majorticklabels():
        tick.set_fontsize(16) 
    for tick in ax.xaxis.get_majorticklabels():
        tick.set_fontsize(16) 

fig.text(0.5, 0, r'Bond Length ($r_e$)', ha='center',fontsize=16)
fig.text(0, 0.5, 'Energy Correlation [mhartree]', va='center', rotation='vertical',fontsize=16)
ax1.legend(fontsize=12,bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=4, mode="expand", borderaxespad=0.)

# Eenerate figure
#plt.show()
plt.savefig('./figure.png',dpi=500,bbox_inches='tight',pad_inches=0)
