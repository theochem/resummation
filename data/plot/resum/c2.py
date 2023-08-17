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
from data.c2_data import *
data = energy
err[:,0] = err[:,0]*1000
degree = []
for i,j in enumerate(err[:,1]):
    degree.append(decdeg2dms(j))
for i in range(6):
    data[i,:] = (data[i,:] - fci)*1000


dim = len(data[0])
w = 0.75
dimw = w / dim
method = ['MPn',r'Pad$\'e$',r'Borel-Pad$\'e$','Meijer-G']


# 创建两个绘图坐标轴；调整两个轴之间的距离，即轴断点距离
fig, axes = plt.subplots(3, 1, sharex=True)
fig.subplots_adjust(hspace=0.1)  # adjust space between axes

# 将用相同的绘图数据，在两个轴上绘制折线图
x = np.arange(len(data))
for i in range(len(data[0])):
    y = [d[i] for d in data]
    # 将用相同的绘图数据，在两个轴上绘制折线图
    if i==2:
        axes[0].bar(x + i * dimw, y, dimw, bottom=0, yerr=err[:,i-2],label=method[i])
        axes[1].bar(x + i * dimw, y, dimw, bottom=0, yerr=err[:,i-2],label=method[i])
        axes[2].bar(x + i * dimw, y, dimw, bottom=0, yerr=err[:,i-2],label=method[i])
    elif i==3:
        p1 = axes[0].bar(x + i * dimw, y, dimw, bottom=0, label=method[i])
        p2 = axes[1].bar(x + i * dimw, y, dimw, bottom=0, label=method[i])
        p3 = axes[2].bar(x + i * dimw, y, dimw, bottom=0, label=method[i])
        axes[0].bar_label(p1, labels=degree, padding=1,fontsize=16)
        axes[1].bar_label(p2, labels=degree, padding=1,fontsize=16)
        axes[2].bar_label(p3, labels=degree, padding=1,fontsize=16)
    else:
        axes[0].bar(x + i * dimw, y, dimw, bottom=0, label=method[i])
        axes[1].bar(x + i * dimw, y, dimw, bottom=0, label=method[i])
        axes[2].bar(x + i * dimw, y, dimw, bottom=0, label=method[i])

# 调整两个y轴的显示范围
axes[0].set_ylim(48000, 50000)  # outliers only
axes[1].set_ylim(-20, 80)  # most of the data
axes[2].set_ylim(-320, -300)  # most of the data


# 创建轴断刻度线，d用于调节其偏转角度
d = 0.5  # proportion of vertical to horizontal extent of the slanted line
kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
              linestyle="none", color='k', mec='k', mew=1, clip_on=False)
axes[0].plot([0, 1], [0, 0], transform=axes[0].transAxes, **kwargs)
axes[1].plot([0, 1], [1, 1], transform=axes[1].transAxes, **kwargs)
axes[1].plot([0, 1], [0, 0], transform=axes[1].transAxes, **kwargs)
axes[2].plot([0, 1], [1, 1], transform=axes[2].transAxes, **kwargs)


x_label = [r'2',r'3',r'4',r'5',r'6',r'20']
axes[-1].set_xticks(x+dimw*1.5)
axes[-1].set_xticklabels(x_label)

# Change the fontsize
for ax in axes:  # example for xaxis
    for tick in ax.yaxis.get_majorticklabels():
        tick.set_fontsize(16) 
    for tick in ax.xaxis.get_majorticklabels():
        tick.set_fontsize(16) 

fig.text(0.5, 0, r'Bond Length ($r_e$)', ha='center',fontsize=16)
fig.text(-0.04, 0.5, 'Energy Correlation [mhartree]', va='center', rotation='vertical',fontsize=16)
axes[0].legend(fontsize=12,bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=4, mode="expand", borderaxespad=0.)

# Eenerate figure
#plt.show()
plt.savefig('./figure.png',dpi=500,bbox_inches='tight',pad_inches=0)
