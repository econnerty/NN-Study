# Erik Connerty
# 6/23/2023
# USC - AI institute

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Defining the network architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, 1000)  # input layer to hidden layer
        self.fc2 = nn.Linear(1000, 1000)  # hidden layer to hidden layer
        self.fc3 = nn.Linear(1000, 1)  # hidden layer to output layer

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

# Check if CUDA is available, else use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create dataset
num_samples = 50000
data = torch.randn([num_samples, 2]).to(device)
labels = (data.norm(dim=1) < 1).float().view(-1, 1).to(device)

# Create network, define loss function and optimizer
net = Net().to(device)
criterion = nn.BCELoss().to(device)  # Binary Cross Entropy Loss
optimizer = torch.optim.SGD(net.parameters(), lr=.1)  # Stochastic Gradient Descent

# Create test data
test_data = torch.randn([10000, 2]).to(device)
test_labels = (test_data.norm(dim=1) < 1).float().view(-1, 1).to(device)

# Training loop
for epoch in range(1000):
    optimizer.zero_grad()  # reset gradients
    outputs = net(data)  # forward pass
    loss = criterion(outputs, labels)  # compute loss
    print(f'Train loss: {loss.item()}')
    loss.backward()  # backpropagation
    optimizer.step()  # update weights
    # Test the trained network
    test_outputs = net(test_data)
    test_loss = criterion(test_outputs, test_labels)
    print(f'Test loss: {test_loss.item()}')
    print('-------------')



# Move the data back to CPU for visualization
test_data_cpu = test_data.cpu()
test_outputs_cpu = test_outputs.detach().cpu()

# Visualize the results
plt.figure(figsize=(8,8))
plt.scatter(test_data_cpu[:, 0], test_data_cpu[:, 1], c=test_outputs_cpu.numpy().ravel(), cmap='coolwarm')
plt.show()
