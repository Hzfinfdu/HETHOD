import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
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
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=12)
x_train=torch.Tensor(x_train)
x_test=torch.Tensor(x_test)
x_train=torch.Tensor(np.concatenate((x_train[:,0:3],x_train[:,5:8],x_train[:,9:11],x_train[:,13:14]),axis=1))
x_test=torch.Tensor(np.concatenate((x_test[:,0:3],x_test[:,5:8],x_test[:,9:11],x_test[:,13:14]),axis=1))

model = LogisticRegression(max_iter=1000)
model.fit(x_train,y_train)
train_score = model.score(x_train,y_train)
test_score = model.score(x_test,y_test)
print('train score: {0}; test score: {1}'.format(train_score,test_score))
