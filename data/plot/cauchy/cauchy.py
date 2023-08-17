import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import math


data = pd.read_excel('./cauchy.csv')
# Initialize the matplotlib figure
fig, axe = plt.subplots()

print(data)

x = data[:]['Distance']
x = [str(i) for i in x]


for (columnName, columnData) in data.loc[:,'MP6':].iteritems():
    y = columnData.values
    plt.plot(x, y, label = columnName)

plt.show()
"""
axe.set_title('MPn')
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

axe.set_xlabel(r'Order',fontsize=15)
axe.set_ylabel('Correlation Energy [mhartree]',fontsize=15)

plt.tight_layout()
plt.savefig(name,dpi=300,bbox_inches='tight',pad_inches=0)
"""
