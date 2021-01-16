import numpy as np
import matplotlib.pyplot as plt
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

class mlpDataset(torch.utils.data.Dataset):
    def __init__(self, train='Train'):
        self.train=train
        if (self.train =='Train'):
            self.x, self.y = x_train,y_train
        else:
            self.x, self.y=x_test,y_test

    def __getitem__(self, index):
            return self.x[index,:], self.y[index]

    def __len__(self):
        if (self.train == 'Train'):
            return 621
        else:
            return 156

class MLP(nn.Module):
    def __init__(self, n_input=9, n_output=2):
        super(MLP, self).__init__()
        self.n_input = n_input
        self.fc_in = nn.Linear(n_input, 256)
        self.S = nn.Sigmoid()
        self.relu=nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.fc_list = nn.ModuleList()
        self.fc_list.append(nn.Linear(256, 32))
        self.fc_list.append(nn.Linear(32, 8))
        self.fc_out = nn.Linear(8, n_output)
        nn.init.uniform_(self.fc_in.weight, -0.1, 0.1)
        nn.init.uniform_(self.fc_out.weight, -0.1, 0.1)
        self.fc_list.apply(self.init_normal)

    def forward(self, x):
        x = self.fc_in(x)
        x = self.relu(x)
        for _, layer in enumerate(self.fc_list, start=0):
            x = self.dropout(layer(x))
            x = self.relu(x)
        x = self.fc_out(x)
        return x

    def init_normal(self, m):
        if type(m) == nn.Linear:
            nn.init.uniform_(m.weight, -0.1, 0.1)

trainset = mlpDataset('Train')
testset = mlpDataset('Test')
if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

trainloader = DataLoader(trainset, batch_size=32, shuffle=False)
testloader = DataLoader(testset, batch_size=32, shuffle=False)

classifier = MLP().to(device)
optimizer = optim.Adam(classifier.parameters(), lr=0.002)
criteria = nn.CrossEntropyLoss()
EPOCHES = 1000
accuracy=[]
num=[]
for epoch in range(EPOCHES):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs_ = inputs.to(device)
        labels_ = labels.to(device)
        optimizer.zero_grad()
        outputs = classifier(inputs_)
        loss = criteria(outputs, labels_.long())
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    if epoch%10==1:
        class_correct = list(0. for i in range(2))
        class_total = list(0. for i in range(2))
        with torch.no_grad():
            for i, data in enumerate(testloader, 0):
                images, labels = data[0].to(device), data[1].to(device)
                outputs = classifier(images)
                _, predicted = torch.max(outputs, 1)
                c = (predicted == labels).squeeze()
                for i in range(28):
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1
        print('\nAccuracy of the network on test: %f %% ' % (100 * sum(class_correct) / sum(class_total)))
        num.append(epoch)
        accuracy.append((100 * sum(class_correct) / sum(class_total)))

plt.plot(num,accuracy)
plt.show()
