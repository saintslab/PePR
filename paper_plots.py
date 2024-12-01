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
from scipy.stats import ttest_ind

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
ms = 50
alpha = 1.0
median_param = np.log10(24.6e6)
### Figure-1

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


### Figure-2
plt.clf()
plt.figure(figsize=(6,5))
plt.subplot(111)
x = np.linspace(0,1,10)
y = np.linspace(0,1,10)

P, E = np.meshgrid(x,y)
PePR = P/(1+E)
plt.contourf(E,P,PePR,levels=10)#,cmap='RdGy')
plt.colorbar(label='PePR-E score');
plt.xlabel('Normalised Energy Unit, $E$')
plt.ylabel('Performance metric, P')
plt.tight_layout()
plt.savefig('results/pepr_profile.pdf',dpi=300)


sns.set_palette('tab10')
allDf = pd.read_csv('data/full_data_pepr_dataset_level.csv')

allDf['cnn'] = allDf.type == 'CNN'
models = pd.read_csv('data/model_names.csv')
models = models.drop_duplicates('model').reset_index()
datasets = ['derma_pt','lidc','lidc_small','derma', \
        'derma_small','derma_smallest','pneumonia','pneumonia_small']

fullDf = pd.read_csv('data/full_data_pepr_dataset_level.csv')

tmp = allDf.drop(columns=['type','dataset']).groupby('model').mean().reset_index()
paramRange = np.linspace(np.log10(0.9*min(tmp.num_param)),np.log10(max(tmp.num_param)),50)

## Models vs param
plt.clf()
plt.figure(figsize=(6,5))
plt.fill_between(paramRange,\
        np.ones(len(paramRange)),\
        where=paramRange  <median_param,
                 facecolor='tab:grey',alpha=0.3)

sns.scatterplot(y=tmp.energy/tmp.energy.max(),x=np.log10(tmp.num_param),hue=models.efficient,style=models.efficient,s=ms)
#plt.plot(median_param*np.ones(10),np.linspace(0,1.02,10),'--',c='tab:red',alpha=0.5)
plt.xlabel('Number of trainable parameters (log$_{10}$)')
plt.ylabel('Energy consumption (kWh)')
plt.grid()
plt.tight_layout()
plt.savefig('results/model_space.pdf',dpi=300)

## Figure 5
### Pretraining influence

plt.clf()
derma_pt = allDf.loc[allDf.dataset=='derma_pt'].reset_index()
derma_npt = allDf.loc[allDf.dataset=='derma'].reset_index()

perf = derma_pt.test_09.values - derma_npt.test_09.values

fig = plt.figure(figsize=(18,6))
gs = fig.add_gridspec(1,6)

ax = fig.add_subplot(gs[0,0])
groupDf = derma_pt[['test_09']]
groupDf['pretrain'] = derma_pt['test_09']
groupDf['no_pretrain'] = derma_npt['test_09']
groupDf = groupDf.drop(columns='test_09')
sns.violinplot(groupDf)
plt.xticks(np.arange(2),['With', 'W/O'])
plt.ylabel('Test Performance')
print(ttest_ind(groupDf.pretrain,groupDf.no_pretrain))
plt.title('a) Pretrain',y=-0.25)
plt.ylim([0.45,0.99])

# statistical annotation
x1, x2 = 0, 1   # columns 'Sat' and 'Sun' (first column: 0, see plt.xticks())
y, h, col = groupDf['pretrain'].max() + 0.06, 0.025, 'k'

plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
plt.text((x1+x2)*.5, y+1.2*h, "p<0.001", ha='center', va='bottom', color=col,fontsize=14)

plt.tight_layout()

ax = fig.add_subplot(gs[0,1])
tmp100 = allDf[(allDf.dataset == 'derma_pt') | (allDf.dataset == 'lidc') | (allDf.dataset == 'pneumonia')].drop(columns=['type','dataset'])
tmp100 = tmp100.groupby('model').mean().reset_index()
tmp10 = allDf[(allDf.dataset == 'derma_small') | (allDf.dataset == 'lidc_small') | (allDf.dataset == 'pneumonia_small')].drop(columns=['type','dataset']).groupby('model').mean().reset_index()

groupDf = derma_pt[['test_09']]

groupDf['train_10'] = tmp10.test_09 
groupDf['train_100'] = tmp100.test_09
print(ttest_ind(groupDf.train_100,groupDf.train_10))


groupDf = groupDf.drop(columns='test_09')
sns.violinplot(groupDf)
plt.xticks(np.arange(2),['10%', '100%'])
plt.ylabel('Test Performance')
plt.title('b) Train set size',y=-0.25)
plt.ylim([0.45,0.99])
# statistical annotation
# Hardcoding the p values based on tests
x1, x2 = 0, 1
y, h, col = groupDf['train_100'].max() + 0.06, 0.025, 'k'

plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
plt.text((x1+x2)*.5, y+1.2*h, "p<0.001", ha='center', va='bottom', color=col,fontsize=14)


plt.tight_layout()

ax = fig.add_subplot(gs[0,2:4])
plt.fill_between(paramRange,\
        np.ones(len(paramRange))*0.85,\
        where=paramRange  <median_param,
                 facecolor='tab:grey',alpha=0.3)

sns.scatterplot(y=tmp.test_09,x=np.log10(tmp.num_param),hue=models.efficient,style=models.efficient,s=ms,alpha=alpha)
#plt.plot(median_param*np.ones(10),np.linspace(0,1.02,10),'--',c='tab:red',alpha=0.5)
plt.grid(axis='y')
plt.ylim([0.3,0.85])
plt.xlabel('Number of trainable parameters (log$_{10}$)')
plt.ylabel('Performance')
plt.title('c) Test performance vs number of parameters',y=-0.25)
plt.tight_layout()

ax = fig.add_subplot(gs[0,4:])
plt.fill_between(paramRange,\
        np.ones(len(paramRange))*0.85,\
        where=paramRange  <median_param,
                 facecolor='tab:grey',alpha=0.3)

sns.scatterplot(y=tmp.pepr_e,x=np.log10(tmp.num_param),hue=models.efficient,style=models.efficient,s=ms)
#plt.plot(median_param*np.ones(10),np.linspace(0,1.02,10),'--',c='tab:red',alpha=0.5)
plt.ylim([0.3,0.85])
plt.grid(axis='y')
plt.xlabel('Number of trainable parameters (log$_{10}$)')
plt.ylabel('PePR-E score')
plt.title('d) PePR-E score vs number of parameters',y=-0.25)


plt.tight_layout()

plt.savefig('results/all_results.pdf',dpi=300)

### Appendix figure

plt.clf()
plt.plot(figsize=(12,5.5))

plt.subplot(131)
plt.fill_between(paramRange,\
        np.ones(len(paramRange))*0.85,\
        where=paramRange  <median_param,
                 facecolor='tab:grey',alpha=0.3)

sns.scatterplot(y=tmp.pepr_c,x=np.log10(tmp.num_param),hue=models.efficient,style=models.efficient,s=ms,alpha=alpha)
#plt.plot(median_param*np.ones(10),np.linspace(0,1.02,10),'--',c='tab:red',alpha=0.5)
plt.grid(axis='y')
plt.ylim([0.3,0.85])
plt.xlabel('Number of trainable parameters (log$_{10}$)')
plt.ylabel('PePR-C score')
plt.title('a) PePR-C score vs num. of parameters',y=-0.25)
plt.tight_layout()

plt.subplot(132)
plt.fill_between(paramRange,\
        np.ones(len(paramRange))*0.85,\
        where=paramRange  <median_param,
                 facecolor='tab:grey',alpha=0.3)

sns.scatterplot(y=tmp.pepr_m,x=np.log10(tmp.num_param),hue=models.efficient,style=models.efficient,s=ms,alpha=alpha)
#plt.plot(median_param*np.ones(10),np.linspace(0,1.02,10),'--',c='tab:red',alpha=0.5)
plt.grid(axis='y')
plt.ylim([0.3,0.85])
plt.xlabel('Number of trainable parameters (log$_{10}$)')
plt.ylabel('PePR-M score')
plt.title('b) PePR-M vs num. of parameters',y=-0.25)
plt.tight_layout()

plt.subplot(133)
plt.fill_between(paramRange,\
        np.ones(len(paramRange))*0.85,\
        where=paramRange  <median_param,
                 facecolor='tab:grey',alpha=0.3)
sns.scatterplot(y=tmp.pepr_t,x=np.log10(tmp.num_param),hue=models.efficient,style=models.efficient,s=ms,alpha=alpha)
#plt.plot(median_param*np.ones(10),np.linspace(0,1.02,10),'--',c='tab:red',alpha=0.5)
plt.grid(axis='y')
plt.ylim([0.3,0.85])
plt.xlabel('Number of trainable parameters (log$_{10}$)')
plt.ylabel('PePR-T score')
plt.title('c) PePR-T score vs num. of parameters',y=-0.25)
plt.tight_layout()

plt.savefig('results/all_pepr.pdf',dpi=300)

