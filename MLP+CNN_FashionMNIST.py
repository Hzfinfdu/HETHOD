import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
import random
import torch.optim as optim
class mlpDataset(torch.utils.data.Dataset):
    def __init__(self, train='Train'):
        self.train=train
        if (self.train =='Train'):
            self.x, self.y = np.load("x_train.npy").astype('float32'), np.load("y_train.npy").astype('int64')
        elif (self.train =='Val'):
            self.x, self.y = np.load("x_train.npy").astype('float32'), np.load("y_train.npy").astype('int64')
        else:
            self.x, self.y = np.load("x_test.npy").astype('float32'), np.load("y_test.npy").astype('int64')
        self.x=torch.Tensor(self.x)
        self.x=self.x.reshape((self.x.shape[0],1,28,28))
        print(self.x.shape)
    def __getitem__(self, index):
        if (self.train!='Val'):
            return self.x[index, :], self.y[index]
        else:
            return self.x[index+50000, :], self.y[index+50000]

    def __len__(self):
        if (self.train == 'Train'):
            return 50000
        else:
            return 10000


class MLP(nn.Module):

    def __init__(self, n_input=784, n_output=10):
        super(MLP, self).__init__()
        self.n_input = n_input
        self.fc_in = nn.Linear(n_input, 1024)
        self.S = nn.Sigmoid()
        self.relu=nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.fc_list = nn.ModuleList()
        self.fc_list.append(nn.Linear(1024, 512))
        self.fc_list.append(nn.Linear(512, 256))
        self.fc_list.append(nn.Linear(256, 64))
        self.fc_out = nn.Linear(64, n_output)
        # Initialize weight
        nn.init.uniform_(self.fc_in.weight, -0.1, 0.1)
        nn.init.uniform_(self.fc_out.weight, -0.1, 0.1)
        self.fc_list.apply(self.init_normal)

    def forward(self, x):
        x = self.fc_in(x)
        x = self.relu(x)
        for _, layer in enumerate(self.fc_list, start=0):
            x=self.dropout(layer(x))
            x=self.relu(x)
        x = self.fc_out(x)
        return x

    def init_normal(self, m):
        if type(m) == nn.Linear:
            nn.init.uniform_(m.weight, -0.1, 0.1)

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1 = nn.Conv2d(1,32,kernel_size=3,stride=1,padding=1)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(32,64,kernel_size=3,stride=1,padding=1)
        self.fc1 = nn.Linear(64*7*7,1024)
        self.fc2 = nn.Linear(1024,512)
        self.fc3 = nn.Linear(512,10)
    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7* 7)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


trainset = mlpDataset('Train')
valset = mlpDataset('Val')
testset = mlpDataset('Test')

if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

trainloader = DataLoader(trainset, batch_size=50, shuffle=False)
valloader = DataLoader(valset, batch_size=50, shuffle=False)
testloader = DataLoader(testset, batch_size=50, shuffle=False)
classifier = CNN().to(device)
optimizer = optim.Adam(classifier.parameters(), lr=0.0002)
criteria = nn.CrossEntropyLoss()
EPOCHES = 10
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
        loss = criteria(outputs, labels_)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 10 == 9:
            print('[Epoch - %d, Batch - %d] loss: %.6f' % (epoch + 1, i + 1, running_loss / 10000))
            running_loss = 0.0
    if epoch%2==1:
        class_correct = list(0. for i in range(10))
        class_total = list(0. for i in range(10))
        with torch.no_grad():
            for i, data in enumerate(testloader, 0):
                images, labels = data[0].to(device), data[1].to(device)
                outputs = classifier(images)
                _, predicted = torch.max(outputs, 1)
                c = (predicted == labels).squeeze()
                for i in range(4):
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1
        print('Accuracy of the network on the 10000 test images: %d %%' % (100 * sum(class_correct) / sum(class_total)))
        num.append(epoch)
        accuracy.append((100 * sum(class_correct) / sum(class_total)))

plt.plot(num,accuracy)
plt.show()
