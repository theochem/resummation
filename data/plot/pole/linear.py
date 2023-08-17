import numpy as np
from resummation.linear_pade import LinearPade
from resummation.linear_borel import LinearBorel
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the mpn energies from the scan path dataset.
data = pd.read_csv('../../data/mpn/b2_scan.csv')
E_mpn = np.array(data[:]['2.6'])
delta_mpn = np.insert(E_mpn[1:] - E_mpn[:-1],0,E_mpn[0])

pade = LinearPade.build(delta_mpn, 23, 23)
bp = LinearBorel.build(delta_mpn, 23, 23)


pade_pole_x = []
pade_pole_y = []
for pole in pade.poles:
    pade_pole_x.append(pole.real)
    pade_pole_y.append(pole.imag)
pade_pole = pd.DataFrame({'x': pade_pole_x,
                          'y': pade_pole_y 
            })



borel_pole_x = []
borel_pole_y = []
for pole in pade.poles:
    print(pole)
for pole in bp.poles:
    borel_pole_x.append(pole.real)
    borel_pole_y.append(pole.imag)
borel_pole = pd.DataFrame({'x': borel_pole_x,
                          'y': borel_pole_y 
            })



sns.set()
# Plot circle of radius 3.
an = np.linspace(0, 2 * np.pi, 100)
fig, axe = plt.subplots()

# 0,0
#sns.scatterplot(ax=axe, data=pade_pole, x='x', y='y')
sns.scatterplot(ax=axe, data=borel_pole, x='x', y='y', color='red')
axe.plot(np.cos(an), np.sin(an),'black')

axe.set_aspect('equal', 'box')
axe.set_xlim(-30, 30)
axe.set_ylim(-30, 30)
axe.set_xlabel('',fontsize=15)
axe.set_ylabel('',fontsize=15)

#plt.show()
plt.savefig('./pole.png',bbox_inches='tight',pad_inches=0)
