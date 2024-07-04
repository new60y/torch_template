import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset,random_split

import numpy as np

import pandas as pd

#import matplotlib.pyplot as plt

import random
import sys

seed =3407
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

model_dir='../model'
data_dir='../data'

class Hyper_para():
    def __init__(self):
       self.batch_size=32 
       self.lr=0.01
       self.L2=0.01
       self.train_rate=0.8
       self.epochs=100
       self.patience=20
       self.device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

args=Hyper_para()

class My_dataset(Dataset):
    def __init__(self):
        pass

    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        feature,label=self.data[idx]
        if label==None:
            return feature
        else:
            #print("label: ",label)
            return feature,label

train_set=My_dataset()

train_len=int(args.train_rate*len(train_set))
valid_len=len(train_set)-train_len
train_set,valid_set=random_split(dataset=train_set,lengths=[train_len,valid_len])
print("train len: ",train_len,"valid len:",valid_len)

train_loader=DataLoader(train_set,shuffle=True,batch_size=args.batch_size)
valid_loader=DataLoader(valid_set,shuffle=True,batch_size=args.batch_size)

class My_module(nn.Module):

    def __init__(self):
        super(My_module,self).__init__()
        
        
    def forward(self,x):
        
        return x

def calc_param(model: nn.Module) -> int:
    params = list(model.parameters())
    param_size = 0
    for _param in params:
        _param_size = 1
        for _ in _param.size():
            _param_size *= _
        param_size += _param_size
    return param_size

model=My_module()
model=model.to(args.device)
print(f"Model parameters: {calc_param(model)}")

def save_check_point(self,model,optimizer,epoch,path): #If needed, lr scheduler should be saved as well
    check_point={
        "net": model.state_dict(),
        'optimizer':optimizer.state_dict(),
        "epoch": epoch
    }
    torch.save(check_point,f'{path}/checkpoint')

#early stop废稿，不用
'''
class Early_stop():
    def __init__(self,patience=20):
        self.patience=patience
        self.no_max_step=0
        self.min_judge=float('inf')

    def __call__(self,judge,model,optimizer,epoch,path):
        if judge<self.min_judge:
            self.min_judge=judge
            self.no_max_step=0
            save_check_point(model,optimizer,epoch,path)
            return False
        else:
            self.no_max_step+=1
            if self.no_max_step>self.patience:
                return True
            else:
                return False
'''

def calc_right(predict,label):
    #print(predict.shape)
    return (predict.argmax(dim=1)==label).sum().item()

def classify(argv):
    criterion=nn.CrossEntropyLoss()
    optimizer=torch.optim.AdamW(
        lr=args.lr,
        eps=1e-8,
        weight_decay=args.L2)

    if '--resume' in argv:
        check_point=torch.load(f'{model_dir}/checkpoint')
        model.load_state_dict(check_point['net'])
        optimizer.load_state_dict(check_point['optimizer'])
        start_epoch = check_point['epoch']+1

    for epoch in range(start_epoch,args.epochs):
        print("epoch",epoch)
        model.train()
        train_acc=0
        train_loss=0
        train_total=0
        for i,(feature,label) in enumerate(train_loader):
            optimizer.zero_grad()
            feature=feature.to(args.device)
            label=label.to(args.device)
            train_total+=label.shape[0]
            predict=model(feature)
            loss=criterion(predict,label)
            train_loss+=loss.item()
            train_acc+=calc_right(predict,label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            #scheduler.step()
        
        train_loss/=len(train_loader)
        train_acc/=train_total
        print("train loss:",train_loss,"train acc:",train_acc)

        valid_loss=0
        valid_acc=0
        valid_total=0
        model.eval()
        with torch.no_grad():
            for i,(feature,label) in enumerate(valid_loader):
                feature=feature.to(args.device)
                label=label.to(args.device)
                valid_total+=label.shape[0]
                predict=model(feature)
                loss=criterion(predict,label)
                valid_loss+=loss.item()
                valid_acc+=calc_right(predict,label)

            valid_loss/=len(valid_loader)
            valid_acc/=valid_total
        print("valid loss:",valid_loss,"valid acc:",valid_acc)

        if max_acc<=valid_acc:
            max_acc=valid_acc
            no_max_step=0
            save_check_point(model,optimizer,epoch,model_dir)
        else:
            no_max_step+=1
            if no_max_step>args.patience:
                break

if __name__=='__main__':
    classify(sys.argv)