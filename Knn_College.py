import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
import heapq
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
import random
import torch.optim as optim
def max(a,b):
    if a>b:
        return a
    else:
        return b

def D(A, B):
    return np.sqrt(sum(np.power((A - B), 2)))

def getListMaxNumIndex(num_list,topk=1):
    num_dict={}
    for i in range(len(num_list)):
        num_dict[i]=num_list[i]
    res_list=sorted(num_dict.items(),key=lambda e:e[1])
    min_num_index=[one[0] for one in res_list[:topk]]
    return min_num_index

x=np.load('x.npy')
y=np.load('y.npy')
Max=0
for i in range(17):
    for j in range(777):
        if x[j][i]>Max:
            Max=x[j][i]
    for j in range(777):
        x[j][i]/=Max
    Max=0
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=2)
x_train=torch.Tensor(x_train)
x_test=torch.Tensor(x_test)
x_train=torch.Tensor(np.concatenate((x_train[:,0:3],x_train[:,5:8],x_train[:,9:11],x_train[:,13:14]),axis=1))
x_test=torch.Tensor(np.concatenate((x_test[:,0:3],x_test[:,5:8],x_test[:,9:11],x_test[:,13:14]),axis=1))

ans=[]
pred=[]
for i in x_test:
    for j in x_train:
        ans.append(D(i,j))
    temp=0
    for k in getListMaxNumIndex(ans):
        if y_train[k]==1:
            temp+=1
    if temp<3:
        pred.append(0)
    else:
        pred.append(1)
    ans.clear()
count=0
for i in range(156):
    if y_test[i]==pred[i]:
        count+=1
print(count/156)
