# Erik Connerty
# 6/23/2023
# USC - AI institute

import torch
from torch import nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# If there's a GPU available...
if torch.cuda.is_available():    
    # Tell PyTorch to use the GPU.    
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

# Define the autoencoder class
class Autoencoder(nn.Module):
    def __init__(self, encoding_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 2048),
            nn.ReLU(True),
            nn.Linear(2048, 2048),
            nn.ReLU(True), 
            nn.Linear(2048, encoding_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 2048),
            nn.ReLU(True),
            nn.Linear(2048, 2048),
            nn.ReLU(True), 
            nn.Linear(2048, 28 * 28), 
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Prepare the MNIST dataset
transform = transforms.Compose([transforms.ToTensor()])
#transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,)),])

trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=128, shuffle=True)

# Initialize the autoencoder and the optimizer
model = Autoencoder(encoding_dim=8).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training loop
for epoch in range(15): 
    for data in trainloader:
        img, _ = data 
        img = img.view(img.size(0), -1)
        img = img.to(device)  # Move the input data to the GPU
        output = model(img)  
        loss = criterion(output, img)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('epoch [{}/{}], loss:{:.4f}'.format(epoch+1, 15, loss.item()))

# Display the reconstructed images after each epoch
with torch.no_grad():
    # Select a batch of images from the training set
    images, _ = next(iter(trainloader))
    images = images.view(images.size(0), -1)
    images = images.to(device)

    # Pass the images through the autoencoder
    reconstructed = model(images)
    reconstructed = reconstructed.view(reconstructed.size(0), 1, 28, 28)
    
    # Move to CPU for displaying
    images = images.cpu()
    reconstructed = reconstructed.cpu()
    
    # Display the first 5 images from the original and reconstructed images
    for i in range(5):
        plt.subplot(2, 5, i+1)
        plt.imshow(images[i].view(28, 28), cmap='gray')
        plt.subplot(2, 5, i+6)
        plt.imshow(reconstructed[i].view(28, 28), cmap='gray')
    plt.show()
