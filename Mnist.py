# 分类实践
# load data
# Bulit Model
# Train
# Test

import torch
import torch.nn as nn
import torchvision 
import numpy as np
from torch.nn import functional as F
from torch import optim
from matplotlib import pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def plot_curve(data):
    fig = plt.figure()
    plt.plot(range(len(data)), data, color = 'blue')
    plt.legend(['value'], loc = 'upper right')
    plt.xlabel('step')
    plt.ylabel('value')
    plt.show()
    
def plot_image(img, label, name):
    
    fig = plt.figure()
    for i in range(6):
        plt.subplot(2,3,i+1)
        plt.tight_layout()
        plt.imshow(img[i][0]*0.3081+0.1307, cmap = 'gray', interpolation='none')
        plt.title('{}: {}'.format(name, label[i].item()))
        plt.xticks([])
        plt.yticks([])
    plt.show()

def one_hot(label, depth=10):
    out = torch.zeros(label.size(0), depth)
    idx = torch.LongTensor(label).view(-1, 1)
    out.scatter_(dim = 1, index=idx, value=1)
    return out

batch_size = 512
# step1
train_dataset = torchvision.datasets.MNIST(root='D:/file', 
                                           train=True, 
                                           transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                                                     torchvision.transforms.Normalize((0.1307,),(0.3081,))]),
                                           download=True)

#正则化过程有助于提高性能

test_dataset = torchvision.datasets.MNIST(root='D:/file', 
                                          train=False, 
                                          transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                                                     torchvision.transforms.Normalize((0.1307,),(0.3081,))]))

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)

x,y = next(iter(train_loader))
print(x.shape, y.shape)
plot_image(x,y, 'image_sample')

class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        
        self.fc1 = nn.Linear(28*28, 256) # 中间层的W和b是需要一些经验性决定的
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 10)
        
    def forward(self, x):
        
        # x : [b, 1, 28, 28]
        # h = relu(Wx+b)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x

net = Net()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
# momentum 是什么概念？
train_loss = []
for epoch in range(5):
    
    for batch_idx, (x,y) in enumerate(train_loader):
        # x: [b, 1, 28, 28], y=[512]
        # [b, 1, 28, 28] -> [b, 784]
        x = x.view(x.size(0), 28*28)
        # -> [b,10]
        out = net(x)
        y_onehot = one_hot(y)
        loss = F.cross_entropy(out, y_onehot)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch_idx % 10 == 0:
            print(epoch, batch_idx, loss.item())
        
        train_loss.append(loss.item())

plot_curve(train_loss)
total_correct = 0
for x,y in test_loader:
    x = x.view(x.size(0), 28*28)
    out = net(x)
    pred = out.argmax(dim = 1)
    correct = pred.eq(y).sum().float().item()
    total_correct += correct

totalnum = len(test_loader.dataset)


acc = total_correct / totalnum

print('test acc:', acc)
for i in range(10):
    x,y = next(iter(train_loader))
    out = net(x.view(x.size(0), 28*28))
    pred = out.argmax(dim = 1)
    plot_image(x, pred, 'test')

