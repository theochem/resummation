import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from b2_data import *

# Initialize the matplotlib figure
plt.style.use(["seaborn-whitegrid", "seaborn-deep", "seaborn-notebook"])
cmap = ['tab:red', 'tab:blue']

# Import data
err = mp48_err*1000
data = mp48
for i in range(5):
    data[i,:] = (data[i,:] - fci[i])*1000
dim = len(data[0])
w = 0.75
dimw = w / dim
method = ['MPn',r'Pad$\'e$',r'Borel-Pad$\'e$']


# 创建两个绘图坐标轴；调整两个轴之间的距离，即轴断点距离
fig, axes = plt.subplots(2, 1, sharex=True)
fig.subplots_adjust(hspace=0.15)  # adjust space between axes


# 将用相同的绘图数据，在两个轴上绘制折线图
x = np.arange(len(data))
for i in range(len(data[0])):
    y = [d[i] for d in data]
    # 将用相同的绘图数据，在两个轴上绘制折线图
    if i==2:
        axes[0].bar(x + i * dimw, y, dimw, bottom=0, yerr=err[:,0],label=method[i])
        axes[1].bar(x + i * dimw, y, dimw, bottom=0, yerr=err[:,0],label=method[i])
    else:
        axes[0].bar(x + i * dimw, y, dimw, bottom=0, label=method[i])
        axes[1].bar(x + i * dimw, y, dimw, bottom=0, label=method[i])

# 调整两个y轴的显示范围
axes[0].set_ylim(10000, 10400)  # outliers only
axes[1].set_ylim(-20, 10)  # most of the data


# 创建轴断刻度线，d用于调节其偏转角度
d = 0.5  # proportion of vertical to horizontal extent of the slanted line
kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
              linestyle="none", color='k', mec='k', mew=1, clip_on=False)
axes[0].plot([0, 1], [0, 0], transform=axes[0].transAxes, **kwargs)
axes[1].plot([0, 1], [1, 1], transform=axes[1].transAxes, **kwargs)


x_label = [r'$1$',r'$1.5$',r'$2.0$',r'$2.6$',r'$3.0$']
axes[1].set_xticks(x+dimw)
axes[1].set_xticklabels(x_label)

# Change the fontsize
for ax in axes:  # example for xaxis
    for tick in ax.yaxis.get_majorticklabels():
        tick.set_fontsize(20) 
    for tick in ax.xaxis.get_majorticklabels():
        tick.set_fontsize(20) 
fig.text(0.5, 0, r'Bond Length ($r_e$)', ha='center',fontsize=20)
fig.text(-0.02, 0.5, 'Energy Correlation [mhartree]', va='center', rotation='vertical',fontsize=20)
axes[0].legend(fontsize=15,bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=4, mode="expand", borderaxespad=0.)

# Eenerate figure
#plt.show()
plt.savefig('./figure.png',dpi=500,bbox_inches='tight',pad_inches=0)
