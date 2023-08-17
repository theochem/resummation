import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import math

file_name = 'b2_mp2.csv'
data = pd.read_csv(file_name)
# Initialize the matplotlib figure
fig, axe = plt.subplots()

x = data[:]['Distance']
x = [str(i) for i in x]


for (columnName, columnData) in data.loc[:,'MPn':].iteritems():
    y = columnData.values
    plt.plot(x, y, marker='o', label = columnName)

plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

axe.set_xlabel(r'Bond Length Multiple',fontsize=15)
axe.set_ylabel('Electron Correlation (%)',fontsize=15)

plt.legend()
plt.tight_layout()
plt.savefig('output.png',dpi=300,bbox_inches='tight',pad_inches=0)

