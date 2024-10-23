import pandas as pd
import numpy as np
import pdb
import seaborn as sns
import matplotlib
import matplotlib._color_data as mcd
cmap = matplotlib.cm.get_cmap('viridis_r')
import matplotlib.pyplot as plt

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


data = pd.read_csv('full_data_pepr_dataset_level.csv')
resnet = pd.read_csv('results-imagenet.csv')
df = data[data.dataset == 'derma_pt']
df.dataset = 'imagenet'
df = df.iloc[:,:5]
df['efficient'] = data.efficient
df['type'] = data.type
df['cnn'] = data.cnn
df['M_n'] = data.M_n
df[['top1','top5']] = 0
for i in range(len(df)):
    idx = np.where(resnet.model.str.contains(df.model[i]))[0]
    if len(idx) > 0:
        idx = idx[0]
        df.iloc[i,-2:] = resnet.iloc[idx,[2,4]].values.tolist()
    else:

        print(df.model[i])

#pdb.set_trace()
df['pepr_m'] = (df.top1/100)/(1+df.M_n)
df.to_csv('full_data_pepr_imagenet.csv',index=False)

### Make pepr-M plot
#pdb.set_trace()
plt.figure(figsize=(12,6))
plt.subplot(121)
sns.scatterplot(y=df.top1/100,x=np.log10(df.num_param),hue=df.type,style=df.efficient,s=ms,alpha=alpha)
plt.grid(axis='y')
plt.ylim([0.4,0.95])
plt.xlabel('Number of trainable parameters (log$_{10}$)')
plt.ylabel('Performance')
plt.title('a) Test performance vs number of parameters',y=-0.25)

plt.subplot(122)
sns.scatterplot(y=df.pepr_m,x=np.log10(df.num_param),hue=df.type,style=df.efficient,s=ms)
plt.ylim([0.3,0.85])
plt.grid(axis='y')
plt.xlabel('Number of trainable parameters (log$_{10}$)')
plt.ylabel('PePR-M score')
plt.title('b) PePR-M score vs number of parameters',y=-0.25)
plt.tight_layout()
plt.savefig('../results/imagenet.pdf',dpi=300)

