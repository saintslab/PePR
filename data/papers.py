import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from cycler import cycler
import numpy as np
import pdb
import matplotlib._color_data as mcd
cmap = matplotlib.cm.get_cmap('viridis_r')
import torch
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

colors = [cmap(i) for i in np.linspace(0,1,11)]

params = {'font.size': 14,
#          'font.weight': 'bold',
          'axes.labelsize':14,
          'axes.titlesize':14,
#          'axes.labelweight':'bold',
          'axes.titleweight':'bold',
          'legend.fontsize': 14,
         }
matplotlib.rcParams.update(params)
#c1 = 'tab:olive'
#c2 = 'tab:green'
ms = 50
alpha = 1.0

df13 = pd.read_csv('data/regions_papers_data_2013.csv')
meanDf13 = df13.groupby('Region_updated').agg({'num_papers':"sum",\
        'pop':'sum'}).reset_index()
meanDf13 = meanDf13.set_index('Region_updated')
meanDf13['pub_pop'] = meanDf13.num_papers/meanDf13['pop']*1e6
meanDf13['year'] = 2013
df23 = pd.read_csv('data/regions_papers_data_2023.csv')
meanDf23 = df23.groupby('Region_updated').agg({'num_papers':"sum",\
        'pop':'sum'}).reset_index()
meanDf23 = meanDf23.set_index('Region_updated')
meanDf23['pub_pop'] = meanDf23.num_papers/meanDf23['pop']*1e6
meanDf23['year'] = 2022

meanDf = pd.concat((meanDf13,meanDf23),axis=0)

plt.figure(figsize=(6,5))
import matplotlib.patches as mpatches

sns.set_palette('viridis')
sns.barplot(meanDf23[['pub_pop']].T,alpha=0.5)
sns.barplot(meanDf13[['pub_pop']].T,alpha=0.95)
#sns.barplot(meanDf.T,hue='year')
plt.xticks(rotation=45,fontsize=10,ha='right',rotation_mode='anchor')
plt.xlabel('')
plt.ylabel('Publications per capita (x $10^6$)')
dark_patch = mpatches.Patch(color='tab:gray', alpha=0.95,label='2013 (dark)')
light_patch = mpatches.Patch(color='tab:gray',alpha=0.5, label='2022 (light)')


plt.legend(handles=[dark_patch,light_patch],fontsize=10)
ax = plt.subplot(111)
ax.spines[['right','top','bottom']].set_visible(False)
plt.tight_layout()
plt.savefig('results/papers.pdf',dpi=300)
