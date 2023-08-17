import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import math



sns.set_theme()

benchmark = pd.read_csv('../data/benchmark/b2_scan.csv')
print(benchmark)

HF = benchmark.at[13,'HF']
FCI = benchmark.at[13,'FCI']
print(HF,FCI)
# Load the brain networks dataset, select subset, and collapse the multi-index
matrix = pd.read_csv('../data/pade/b2/b2_2_6.csv')
print(matrix)
#matrix.drop(matrix.columns[0], inplace=True, axis=1)
matrix = (matrix.iloc[0:24,0:24]-FCI)*1000
matrix = matrix[abs(matrix[:][:])<100]
print(matrix)
matrix.index = map(str,range(1,25))
matrix.columns = map(str,range(1,25))
matrix.index.name = 'Denominator'

matrix = matrix.reset_index()
mydata = pd.melt(matrix, id_vars='Denominator', var_name='Numerator', value_name='value')
mydata['AbsValue'] = np.abs(mydata.value)

# Draw each cell as a scatter point with varying size and color
g = sns.scatterplot(
    data=mydata,
    x="Denominator", y="Numerator",
    hue="value", size="AbsValue",
    #size_norm=(0, 100), 
    sizes=(20, 100),
    palette="viridis",
    #palette="RdYlGn",
)

h,l = g.get_legend_handles_labels()
l = l[l.index('value')+1:l.index('AbsValue')]
l = ['EC']+l

def legend(handel,label):
    l = copy.deepcopy(label)
    new_h,new_l = [],[]

    label = label[label.index('value')+1:label.index('AbsValue')]
    label = list(map(float,label))
    label = list(map(lambda x : round(x/10)*10,label))
    set_list = list(dict.fromkeys(label))
    label = ['EC']+label
    index = [0]
    for i in set_list:
        index.append(label.index(i))
    for i in index:
        new_h.append(handel[i])
        new_l.append(label[i])

    return new_h,new_l

#new_h,new_l = legend(h,l)
plt.legend(h,l,bbox_to_anchor=(1.25, 1.0))
plt.tight_layout()
plt.savefig('fig.png',dpi=400)
#plt.show()
