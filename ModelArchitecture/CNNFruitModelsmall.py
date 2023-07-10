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
        self.conv1 = nn.Conv2d(3, 6, (10,10), padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.LazyLinear(256)  # 64*8*8 is the flattened volume before FC layers
        self.fc2 = nn.Linear(256, 30)  # output nodes = number of classes
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = torch.flatten(x,1)  # Flatten the tensor
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x