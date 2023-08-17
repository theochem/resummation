import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import math

font = {'family' : 'normal',
        'size'   : 50}

mpl.rc('font', **font)


benchmark = pd.read_csv('../../data/benchmark/b2_scan.csv')

mpn_data = pd.read_csv('../../data/mpn/b2_scan.csv')

def single_plot(HF,FCI, x, y,name):
    # Initialize the matplotlib figure
    #plt.style.use('seaborn')
    plt.style.use(["seaborn-whitegrid", "seaborn-deep", "seaborn-notebook"])

    cmap = ['tab:red', 'tab:blue']
    fig, axe = plt.subplots()

    y = (y-FCI)*1000
    y = list(y)

    c = [cmap[int(i>=0)] for i in y]
    abs_y = [abs(i) for i in y]

    axe.bar(x=x, height=abs_y,color=c)
    #axe.set_title('MPn')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    axe.set_xlabel(r'Order',fontsize=15)
    axe.set_ylabel('Correlation Energy [mhartree]',fontsize=15)
    axe.set_ylim([0,100])

    plt.tight_layout()
    plt.savefig(name,dpi=300,bbox_inches='tight',pad_inches=0)

def scan(benchmark,mpn_data,name):
    for i,d in enumerate(mpn_data[:]):
        x = range(2,49)
        y = mpn_data[:][d][2:]
        HF = benchmark.at[i,'HF']
        FCI = benchmark.at[i,'FCI']
        single_plot(HF,FCI,x,y,name+'_'+str(i))

#x = [str(i) for i in range(2,21)]
#single_plot(-26.39079696,-26.50889068, x, mpn_data, 'name')
scan(benchmark,mpn_data,'b2')

