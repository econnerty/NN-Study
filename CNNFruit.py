# Erik Connerty
# 6/23/2023
# USC - AI institute

import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from PIL import Image
import tkinter as tk
from tkinter import filedialog
import os

from Models.CNNFruitModel import Net

#Define the transform for the images
transform = transforms.Compose([
    transforms.Resize((128,128)),    # Resize images to 64x64
    transforms.ToTensor(),         # Convert image to PyTorch Tensor data type
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])   # Normalize
])

def Train():
    # Check if GPU is available and if not, fall back on CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data from folders
    data = datasets.ImageFolder(root='./Data/Fruit', transform=transform)
    test_data = datasets.ImageFolder(root='./Data/FruitTest', transform=transform)
    # Force the class-to-index mappings to be the same
    test_data.class_to_idx = data.class_to_idx

    # Split data into train and test sets
    #train_size = int(.99 * len(data))  # Use 80% of the data for training
    #test_size = len(data) - train_size
    #train_data, test_data = torch.utils.data.random_split(data, [train_size, test_size])
    #train_data = data
    # Create data loaders
    trainloader = DataLoader(data, batch_size=256, shuffle=True,pin_memory=True,num_workers=8)
    testloader = DataLoader(test_data, batch_size=8, shuffle=False)


    # Create an instance of the model
    model = Net()
    # Load model if it exists
    if os.path.isfile('model.pt'):
        model.load_state_dict(torch.load('model.pt'))
        print('Model loaded')
    model.to(device)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

  # Train the model
    for epoch in range(100):  # loop over the dataset multiple times
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(trainloader)

        model.eval()
        running_test_loss = 0.0
        for inputs, labels in tqdm(testloader):
            inputs, labels = inputs.to(device), labels.to(device)
            with torch.no_grad():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            running_test_loss += loss.item()

        test_loss = running_test_loss / len(testloader)

        print(f'\nEpoch {epoch+1}, Train Loss: {train_loss}, Test Loss: {test_loss}')
            

    print('Finished Training')

    # Save the trained model
    torch.save(model.state_dict(), 'model.pt')
    
def Test():
    # Load data from folders
    data = datasets.ImageFolder(root='./Data/Fruit', transform=transform)

    # Create a dictionary that maps indices to class names
    idx_to_class = {v: k for k, v in data.class_to_idx.items()}
    # Define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create an instance of the model
    model = Net()
    model.load_state_dict(torch.load('model.pt'))    # Load the trained model
    model = model.to(device)
    model.eval()    # Set the model to evaluation mode

    # Function for image classification
    def classify_image():
        root = tk.Tk()
        root.withdraw()    # we don't want a full GUI, so keep the root window from appearing

        # Show an "Open" dialog box and return the path to the selected file
        file_path = filedialog.askopenfilename()
        image = Image.open(file_path)

        # Apply the transformations to the image and add a batch dimension
        image = transform(image).unsqueeze(0).to(device)

        # Forward pass the image through the model
        output = model(image)

        # Get the index of the highest probability class
        _, predicted_class = torch.max(output, 1)
        
        # Map the predicted class index to the class name
        predicted_class_name = idx_to_class[predicted_class.item()]

        print("Predicted class: ", predicted_class_name)

    # Classify an image
    while True:
        classify_image()
if __name__ == '__main__':
    while True:
        print('Press 1 to train and press 2 to test')
        user_input = input()
        if user_input == '1':
            Train()
        elif user_input == '2':
            Test()
        else:
            print('Invalid input')
    

