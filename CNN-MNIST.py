# Erik Connerty
# 6/23/2023
# USC - AI institute

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# Check if GPU is available and if not, fall back on CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the CNN architecture
class Net(nn.Module):   
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)  # Input is 1 (grayscale image), output is 32, kernel size is 3, stride is 1.
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.LazyLinear(64*12*12, 1024)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = torch.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = torch.log_softmax(x, dim=1)
        return output

# Define a transform to normalize the data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
if __name__ == '__main__':
    # Load the MNIST train and test datasets
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # Initialize a loader for the training data
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True,pin_memory=True)

    # Initialize a loader for the testing data
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False,pin_memory=True)

    # Initialize the network and optimizer
    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # Specify the loss function
    criterion = nn.CrossEntropyLoss().to(device)

    # Train the network
    for epoch in range(10):  # loop over the dataset multiple times

        model.train()
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
        train_loss = running_loss / len(trainloader)


        model.eval()
        running_test_loss = 0.0
        with torch.no_grad():
            for i, data in enumerate(testloader, 0):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_test_loss += loss.item()
        test_loss = running_test_loss / len(testloader)
        print(f'\nEpoch {epoch+1}, Train Loss: {train_loss}')
        print(f'Epoch {epoch+1}, Test Loss: {test_loss}')

    print('Finished Training')
