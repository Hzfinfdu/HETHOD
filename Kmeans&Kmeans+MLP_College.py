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

x=np.load('x.npy')
y=np.load('y.npy')
max=0
for i in range(17):
    for j in range(777):
        if x[j][i]>max:
            max=x[j][i]
    for j in range(777):
        x[j][i]/=max
    max=0
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=12)
x_train=torch.Tensor(x_train)
x_test=torch.Tensor(x_test)
x_train=torch.Tensor(np.concatenate((x_train[:,0:3],x_train[:,5:8],x_train[:,9:11],x_train[:,13:14]),axis=1))
x_test=torch.Tensor(np.concatenate((x_test[:,0:3],x_test[:,5:8],x_test[:,9:11],x_test[:,13:14]),axis=1))

def max(a,b):
    if a>b:
        return a
    else:
        return b

def D(A,B):
    return np.sqrt(sum(np.power((A - B), 2)))

class KMeans:
    def __init__(self, n_clusters=10, max_iter=None, verbose=False, device=torch.device("cpu")):
        self.n_clusters = n_clusters
        self.labels = None
        self.dists = None
        self.centers = None
        self.variance = torch.Tensor([float("inf")]).to(device)
        self.verbose = verbose
        self.started = False
        self.max_iter = max_iter
        self.count = 0
        self.device = device

    def fit_predict(self, x):
        init_row = torch.randint(0, x.shape[0], (self.n_clusters,)).to(self.device)  # torch.randint(low=0, high, size,)
        init_points = x[init_row]
        self.centers = init_points
        while True:
            self.nearest_center(x)
            self.update_center(x)
            if self.verbose:
                print(self.variance, torch.argmin(self.dists,(0)))
            if torch.abs(self.variance) < 1e-5 and self.max_iter is None:
                break
            elif self.max_iter is not None and self.count == self.max_iter:
                break
            self.count += 1
        self.representive_sample()
        return self.labels

    def nearest_center(self, x):
        labels = torch.empty((x.shape[0],)).to(self.device)
        dists = torch.empty((0, self.n_clusters)).to(self.device)
        for i, sample in enumerate(x):
            dist = torch.sum(torch.mul(sample - self.centers, sample - self.centers),(1))
            labels[i] = torch.argmin(dist)
            dists = torch.cat([dists, dist.unsqueeze(0).to(self.device)], 0)
        self.labels = labels
        if self.started:
            self.variance = torch.sum(self.dists - dists)
        self.dists = dists
        self.started = True

    def update_center(self, x):
        centers = torch.empty((0, x.shape[1])).to(self.device)
        for i in range(self.n_clusters):
            mask = (self.labels == i)
            cluster_samples = x[mask]
            centers = torch.cat([centers,torch.mean(cluster_samples,0).unsqueeze(0)],0)
        self.centers = centers
    def representive_sample(self):
        self.representative_samples = torch.argmin(self.dists,(0))
        return self.representative_samples

'''xx=[]
yy=[]
for i in range(17):
    xx.append(i+1)
    x_train_t=torch.Tensor(np.concatenate((x_train[:,i:i+1],x_train[:,i:i+1]),axis=1))
    x_test_t=torch.Tensor(np.concatenate((x_test[:,i:i+1], x_test[:,i:i+1]),axis=1))
    kmeans=KMeans(n_clusters=2)
    kmeans.fit_predict(x_train_t)
    r1=y_train[kmeans.representative_samples[0]]
    ans=[]
    for i in x_test_t:
        if D(i,kmeans.centers[0])>D(i,kmeans.centers[1]):
            ans.append(1-r1)
        else:
            ans.append(r1)
    count=0
    for i in range(len(ans)):
        if ans[i]==y_test[i]:
            count+=1
    yy.append(max(1-count/len(ans),count/len(ans)))
plt.plot(xx,yy)
plt.show()'''

kmeans=KMeans(n_clusters=2)
kmeans.fit_predict(x_train)
r1=y_train[kmeans.representative_samples[0]]
ans=[]
for i in x_test:
    if D(i,kmeans.centers[0])<D(i,kmeans.centers[1]):
        ans.append(r1)
    else:
        ans.append(1-r1)
count=0
for i in range(len(ans)):
    if ans[i]==y_test[i]:
        count+=1
print(count/len(ans))

kmeans=KMeans(n_clusters=2)
kmeans.fit_predict(x_train)
bias=torch.Tensor(np.concatenate((x_train[kmeans.representative_samples[0]],x_train[kmeans.representative_samples[1]])))
x_train=torch.Tensor(np.concatenate((x_train,x_train),axis=1))
x_test=torch.Tensor(np.concatenate((x_test,x_test),axis=1))
for i in x_train:
    i-=bias
for i in x_test:
    i-=bias

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
    def __init__(self, n_input=18, n_output=2):
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
    if epoch%100==1:
        class_correct = list(0. for i in range(10))
        class_total = list(0. for i in range(10))
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
