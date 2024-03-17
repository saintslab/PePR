import pandas as pd
import numpy as np
import pdb

allDf = pd.read_csv('data/all_results.csv')

allDf['cnn'] = allDf.type == 'CNN'
models = pd.read_csv('data/model_names.csv')
models = models.drop_duplicates('model').reset_index()
datasets = ['derma_pt','lidc','lidc_small','derma', \
        'derma_small','derma_smallest','pneumonia','pneumonia_small']
dataset_pairs = [['derma_pt','derma_small'], \
        ['lidc','lidc_small'], \
        ['pneumonia','pneumonia_small']]

# PePR-E per experiment (dataset)
i = 0
#Emax = allDf.energy.max()*1000
#Emin = allDf.energy.min()*1000
for d in datasets:

    singleDf = allDf[allDf.dataset==d]
    
    # Treat every converged model after as a model in the experiment space
    Emax = singleDf.energy.max() # Use the largest training energy across epochs
    Emin = singleDf.energy.min() # Use the smallest training energy per epoch
    En = (singleDf.energy.values - Emin)/(Emax-Emin)

    print(En.max(),En.min(),d)
#    pdb.set_trace()
    singleDf.loc[:,'E_n'] = En
    singleDf.loc[:,'pepr_e'] = singleDf.test_09.values/(1+En)
#    singleDf.loc[:,'pepr_e'] = singleDf.best_test.values/(1+En)


    # PePR-C
    Cmax = singleDf.co2.max() # Use the largest training energy across epochs
    Cmin = singleDf.co2.min() # Use the smallest training energy per epoch

    Cn = (singleDf.co2.values - Cmin)/(Cmax-Cmin)

    singleDf.loc[:,'C_n'] = Cn
    singleDf.loc[:,'pepr_c'] = singleDf.test_09.values/(1+Cn)

    # PePR-T
    Tmax = singleDf.train_time.max() # Use the largest training energy across epochs
    Tmin = singleDf.train_time.min() # Use the smallest training energy per epoch

    Tn = (singleDf.train_time.values - Tmin)/(Tmax-Tmin)

    singleDf.loc[:,'T_n'] = Tn
    singleDf.loc[:,'pepr_t'] = singleDf.test_09.values/(1+Tn)

    # PePR-M
    Mmax = singleDf.memR.max() # Use the largest training energy across epochs
    Mmin = singleDf.memR.min() # Use the smallest training energy per epoch

    Mn = (singleDf.memR.values - Mmin)/(Mmax-Mmin)

    singleDf.loc[:,'M_n'] = Mn
    singleDf.loc[:,'pepr_m'] = singleDf.test_09.values/(1+Mn)

    if i == 0:
        fullDf = singleDf
        i+=1
    else:
        fullDf = pd.concat((fullDf,singleDf),axis=0)
fullDf.to_csv('data/full_data_pepr_dataset_level.csv',index=False)

