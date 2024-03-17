import timm
import torch
from torchvision import transforms, datasets
from torch.utils.data import SubsetRandomSampler,random_split, DataLoader
import numpy as np
import pandas as pd
from data.datasets import LIDCdataset, DermaMNIST
from tools import makeLogFile,writeLog,dice,dice_loss,multiClassAccuracy
import time
import pdb
from carbontracker.tracker import CarbonTracker
from carbontracker import parser as ctparser
import os
import shutil
import argparse

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

torch.manual_seed(1)
# Globally load device identifier
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
mFile = 'model_results.csv'

def evaluate(loader):
    ### Evaluation function for validation/testing
    vl_acc = torch.Tensor([0.]).to(device)
    vl_loss = 0.
    labelsNp = []
    predsNp = []
    model.eval()

    for i, (inputs, labels) in enumerate(loader):
        b = inputs.shape[0]
        inputs = inputs.to(device)
        labels = labels.to(device)
        # Inference
        scores = model(inputs)
#        scores = scores.view(labels.shape).type_as(labels)

        preds = torch.sigmoid(scores.clone())
        loss = loss_fun(scores, labels.squeeze())
        vl_loss += loss
        vl_acc += accuracy(labels,preds)

    # Compute AUC over the full (valid/test) set
    vl_acc = vl_acc.item()/len(loader)
    vl_loss = vl_loss.item()/len(loader)

    return vl_acc, vl_loss

# Load dataset
train_set = DermaMNIST(split='train')
valid_set = DermaMNIST(split='valid')
test_set = DermaMNIST(split='test')

x = train_set[0][0]
dim = x.shape[-1]
nCh = x.shape[0]
print('Using %d size of images'%dim)
N = len(train_set)

# Initiliaze input dimensions
num_train = len(train_set)
num_valid = len(valid_set)
num_test = len(test_set)
print("Num. train = %d, Num. val = %d, Num. test = %d"%(num_train,num_valid,num_test))

batch_size = 32
# Assign script args to vars

# Initialize dataloaders
loader_train = DataLoader(dataset = train_set, drop_last=False,num_workers=0,
        batch_size=batch_size, pin_memory=True)
loader_valid = DataLoader(dataset = valid_set, drop_last=True,num_workers=0,
        batch_size=batch_size, pin_memory=True)
loader_test = DataLoader(dataset = test_set, drop_last=True,num_workers=0,
        batch_size=batch_size, pin_memory=True)

nValid = len(loader_valid)
nTrain = len(loader_train)
nTest = len(loader_test)

# Initialize loss and metrics
loss_fun = torch.nn.CrossEntropyLoss()
accuracy = multiClassAccuracy
num_epochs = 10
lr = 5e-4

df = pd.read_csv(mFile)

models = df.model.values

parser = argparse.ArgumentParser()
parser.add_argument('--mIdx', type=int, default=0, help='Model Index')
args = parser.parse_args()
mIdx = args.mIdx

if df.loc[mIdx,'memT'] == 0:

    print('Using ',models[mIdx])

    model = timm.create_model(models[mIdx],pretrained=True,in_chans=nCh,num_classes=7)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    torch.cuda.empty_cache()
    model.to(device)


    # Instantiate Carbontracker
    logLoc = './logs/'+models[mIdx]
    if not os.path.exists(logLoc):
        os.mkdir(logLoc)
    tracker = CarbonTracker(epochs=num_epochs,
            log_dir=logLoc,monitor_epochs=-1)
    infTime = []
    # Miscellaneous initialization
    start_time = time.time()

    # Training starts here
    for epoch in range(num_epochs):
        tracker.epoch_start()
        running_loss = 0.
        running_acc = 0.
        t = time.time()
        model.train()
        predsNp = []
        labelsNp = []
        bNum = 0
        for i, (inputs, labels) in enumerate(loader_train):


            inputs = inputs.to(device)
            labels = labels.to(device)

            for p in model.parameters():
                p.grad = None
            bNum += 1
            b = inputs.shape[0]

    #        pdb.set_trace()
            scores = model(inputs)
    #        scores = scores.view(labels.shape).type_as(labels)
            loss = loss_fun(scores, labels.squeeze())

            # Backpropagate and update parameters
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                preds = torch.sigmoid(scores.clone())
                running_acc += (accuracy(labels,preds)).item()
                running_loss += loss.item()

            if (i+1) % 10 == 1:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                       .format(epoch+1, num_epochs, i+1, nTrain, loss.item()))

        tr_acc = running_acc/nTrain

        if epoch == 1:
            df.loc[mIdx,'memT'] = torch.cuda.get_device_properties(0).total_memory/1e9
            df.loc[mIdx,'memR'] = torch.cuda.memory_reserved(0)/1e9
            df.loc[mIdx,'memA'] = torch.cuda.memory_allocated(0)/1e9
            df.loc[mIdx,'memM'] = torch.cuda.max_memory_allocated(0)/1e9

        # Evaluate on Validation set
        with torch.no_grad():

            iTime = time.time()
            best_ts_acc, best_ts_loss = evaluate(loader=loader_test)
            infTime.append(time.time()-iTime)
            df.loc[mIdx,'test_%02d'%epoch] = best_ts_acc
            print('Test Set Loss:%.4f\t Acc:%.4f'%(best_ts_loss, best_ts_acc))
        tracker.epoch_end()
    tracker.stop()
    df.loc[mIdx,'train_time'] = time.time()-t
    df.loc[mIdx,'inf_time'] = np.mean(infTime)
    df.loc[mIdx,'tr_acc'] = tr_acc
    #pdb.set_trace()
    log = ctparser.parse_all_logs(log_dir=logLoc)
    df.loc[mIdx,'energy'] = log[-1]['actual']['energy (kWh)'] 
    df.loc[mIdx,'co2'] = log[-1]['actual']['co2eq (g)'] 

    df.to_csv(mFile,index=False)
    #torch.save(model.state_dict(),'models/'+models[mIdx]+'.pt')
