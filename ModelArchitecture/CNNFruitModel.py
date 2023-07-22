# Erik Connerty
# 6/23/2023
# USC - AI institute

import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# Define the CNN
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, (15,15), padding=1)
        self.conv2 = nn.Conv2d(16, 32, (4,4), padding=1)  # input channels = 8, output channels=16
        self.pool = nn.MaxPool2d((10,10), 2)
        self.pool2 = nn.MaxPool2d((3,3), 2)
        self.fc1 = nn.LazyLinear(512)  # 64*8*8 is the flattened volume before FC layers
        self.fc2 = nn.Linear(512, 30)  # output nodes = number of classes
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = torch.flatten(x,1)  # Flatten the tensor
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.softmax(self.fc2(x))
        return x