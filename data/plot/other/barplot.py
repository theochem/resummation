import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import math


# Initialize the matplotlib figure
plt.style.use('seaborn')
cmap = ['tab:red', 'tab:blue']
fig, axes = plt.subplots(2, 2)

benchmark = pd.read_csv('../data/benchmark/lih_scan.csv')
HF = benchmark.at[16,'HF']
FCI = benchmark.at[16,'FCI']

data = pd.read_csv('../data/barplot/lih_2_8.csv')


# 0,0
x_00 = [r'$\frac{1}{1}$',r'$\frac{1}{2}$',r'$\frac{2}{2}$',r'$\frac{2}{3}$',r'$\frac{3}{3}$',
    r'$\frac{3}{4}$',r'$\frac{4}{4}$',r'$\frac{4}{5}$',r'$\frac{5}{5}$',
    r'$\frac{5}{6}$',r'$\frac{6}{6}$',r'$\frac{6}{7}$',r'$\frac{7}{7}$',
    r'$\frac{7}{8}$',r'$\frac{8}{8}$',r'$\frac{8}{9}$',r'$\frac{9}{9}$',
    r'$\frac{9}{10}$',r'$\frac{10}{10}$']

y_00 = data[:]['pade']
y_00 = (HF-y_00)/(HF-FCI)*100-100
y_00 = list(y_00)
c_00 = [cmap[int(i>=0)] for i in y_00]
abs_y_00 = [abs(i) for i in y_00]

axes[0,0].bar(x=x_00, height=abs_y_00,color=c_00)
axes[0,0].set_title(r'$Pad\'e$')
axes[0,0].set(xlabel=r'Order')
axes[0,0].set(ylabel=r'Percent Error $(\%)$')
axes[0,0].set_ylim([0,100])

# 0,1
x_01 = x_00

y_01 = data[:]['borel']
y_01 = (HF-y_01)/(HF-FCI)*100-100
y_01 = list(y_01)

yerr_01 = data[:]['berr']
yerr_01 = yerr_01/(HF-FCI)*100
yerr_01 = list(yerr_01)

c_01 = [cmap[int(i>=0)] for i in y_01]
abs_y_01 = [abs(i) for i in y_01]

axes[0,1].bar(x=x_01, height=abs_y_01,yerr=yerr_01,color=c_01)
axes[0,1].set_title(r'Borel $Pad\'e$')
axes[0,1].set(xlabel=r'Order')
axes[0,1].set(ylabel=r'Percent Error $(\%)$')
axes[0,1].set_ylim([0,100])

# 1,0

x_10 = [str(i) for i in range(2,21)]

y_10 = data[:]['mpn']
y_10 = (HF-y_10)/(HF-FCI)*100-100
y_10 = list(y_10)

c_10 = [cmap[int(i>=0)] for i in y_10]
abs_y_10 = [abs(i) for i in y_10]

axes[1,0].bar(x=x_10, height=abs_y_10,color=c_10)
axes[1,0].set_title('MPn')
axes[1,0].set(xlabel=r'Order')
axes[1,0].set(ylabel=r'Percent Error $(\%)$')
axes[1,0].set_ylim([0,100])

# 1,1

x_11 = [str(i) for i in range(1,20)]

y_11 = data[:]['meijer']
y_11 = (HF-y_11)/(HF-FCI)*100-100
y_11 = list(y_11)

yerr_11 = data[:]['merr']
yerr_11 = yerr_11/(HF-FCI)*100
yerr_11 = list(yerr_11)
c_11 = [cmap[int(i>=0)] for i in y_11]
abs_y_11 = [abs(i) for i in y_11]

axes[1,1].bar(x=x_11, height=abs_y_11,color=c_11,yerr=yerr_11)
axes[1,1].set_title('Meijer G')
axes[1,1].set(xlabel=r'Order')
axes[1,1].set(ylabel=r'Percent Error $(\%)$')
axes[1,1].set_ylim([0,100])

plt.tight_layout()
plt.show()
