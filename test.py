
import torch
import torchvision
import matplotlib.pyplot as plt
import pandas
import ignite
import torch.optim as optim
from torch import nn

import torch.nn.functional as F
from torchvision import transforms, datasets
import torch.backends.mkl as M
import torch.backends.mkldnn as E


# downloading mnist
train_dl = datasets.MNIST("", train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
test_dl = datasets.MNIST("", train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))
# loading mnist with dataloader
trainset = torch.utils.data.DataLoader(train_dl, batch_size=10, shuffle=True)
testset = torch.utils.data.DataLoader(test_dl, batch_size=10, shuffle=True)


# creating network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


net = Net()
print(net)
loss = nn.CrossEntropyLoss()  # loss function using cross entropy
optimizer = optim.Adam(net.parameters(), lr=0.001)  # optimizer function
import cv2
import numpy as np


img=cv2.imread("/home/danhyal/test.jpg")
cap=cv2.VideoCapture(0)
det,frame=cap.read()

frame=cv2.cv2.cvtColor(frame,cv2.cv2.COLOR_BGR2RGBA)
frame=np.array(frame).dot(frame)
imgplot=plt.imshow(frame)

plt.show()

def forwardpass():
    for epoch in range(9):
        for indx, data in enumerate(trainset):
            X, y = data
            net.zero_grad()
            output = net(X)
            loss = F.nll_loss(output, y)
            loss.backward()
            optimizer.step()
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, indx * len(data), len(trainset.dataset),
                100. * indx / len(trainset), loss.item()))



