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
        self.fc1 = nn.LazyLinear(512)  # 64*8*8 is the flattened volume before FC layers
        self.fc3 = nn.Linear(512, 30)  # output nodes = number of classes
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = torch.flatten(x,1)  # Flatten the tensor
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x