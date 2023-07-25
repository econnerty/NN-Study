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
from torch.utils.tensorboard import SummaryWriter

import ModelArchitecture.CNNFruitModel as CNNFruitModel
import ModelArchitecture.CNNFruitModelsmall as CNNFruitModelSmall
import ModelArchitecture.CNNFruitModelnocnnlarge as CNNFruitModelNoCNNLarge
import ModelArchitecture.CNNFruitModelnocnnsmall as CNNFruitModelNoCNNSmall
import ModelArchitecture.CNNFruitModelnew as CNNFruitModelnew

#Define the transform for the images
transform = transforms.Compose([
    transforms.Resize((128,128)),    # Resize images to 128x128
    transforms.ToTensor(),         # Convert image to PyTorch Tensor data type
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])   # Normalize the rgb values
])
hparams = {
    'lr': 0.003,
    'batch_size': 64,
    'num_epochs': 10,
    'weight_decay': 1e-6
}

    # Create an instance of the model
    #model = Net()
    # Load model if it exists
    #if os.path.isfile('./TrainedModels/fruitmodel.pt'):
    #    model.load_state_dict(torch.load('./TrainedModels/fruitmodel.pt'))
    #    print('Model loaded')

import csv

def Train(hparams, run_name, model):
    # Check if GPU is available and if not, fall back on CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data from folders
    data = datasets.ImageFolder(root='./Data/Fruit', transform=transform)
    test_data = datasets.ImageFolder(root='./Data/FruitTest', transform=transform)
    # Force the class-to-index mappings to be the same
    test_data.class_to_idx = data.class_to_idx

    # Create data loaders
    trainloader = DataLoader(data, batch_size=hparams['batch_size'], shuffle=True,pin_memory=True,num_workers=6)
    testloader = DataLoader(test_data, batch_size=1, shuffle=False)
    
    model.to(device)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=hparams['lr'],weight_decay=hparams['weight_decay'])
    
    #Initialize tensorboard
    writer = SummaryWriter(f'./logs/{run_name}')


    # Create CSV log files
    with open(f'./logs/{run_name}_train.csv', 'w', newline='') as file:
        writer_train = csv.writer(file)
        writer_train.writerow(["Epoch", "Batch", "Training Loss"])

    with open(f'./logs/{run_name}_test.csv', 'w', newline='') as file:
        writer_test = csv.writer(file)
        writer_test.writerow(["Epoch", "Test Loss"])
    # Train the model
    model.train()
    for epoch in range(hparams['num_epochs']):  # loop over the dataset multiple times
        running_loss = 0.0
        for i,(inputs, labels) in enumerate(tqdm(trainloader)):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            # Log the loss
            writer.add_scalar('Training Loss', loss.item(), epoch*len(trainloader) + i)
            # Log the learning rate
            #writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch*len(trainloader) + i)
            # Write the loss and learning rate to the CSV file
            with open(f'./logs/{run_name}_train.csv', 'a', newline='') as file:
                writer_train = csv.writer(file)
                writer_train.writerow([epoch, i, loss.item()])

        train_loss = running_loss / len(trainloader)
        print(f'\nEpoch {epoch+1}, Train Loss: {train_loss}')

        # Evaluation loop (for test loss)
        model.eval()  # set model to evaluation mode
        test_loss = 0.0
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            with torch.no_grad():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            test_loss += loss.item()
        test_loss = test_loss / len(testloader)  # calculate average test loss over an epoch
        writer.add_scalar('Test Loss', test_loss, epoch)  # log test loss to TensorBoard
        # Write the test loss to the CSV file
        with open(f'./logs/{run_name}_test.csv', 'a', newline='') as file:
            writer_test = csv.writer(file)
            writer_test.writerow([epoch, test_loss])
        
        model.train()  # set model back to training mode

    # Save the model
    torch.save(model.state_dict(), f'./TrainedModels/fruitmodel_{run_name}.pt')
    print('Finished Training')
    # Export the model
    testinputs, classes = next(iter(trainloader)) 
    testinputs = testinputs.to(device)  
    torch.onnx.export(model, testinputs, f"./TrainedModels/fruitmodel-vis_epoch{run_name}.onnx")
    
    # Add the hparams to TensorBoard
    writer.add_hparams(hparams, {'hparam/num_parameters': sum(p.numel() for p in model.parameters())})
    
    # Get total parameter count
    total_params = sum(p.numel() for p in model.parameters())
    
    # Create CSV log files
    with open(f'./logs/{run_name}_params.csv', 'w', newline='') as file:
        writer_train = csv.writer(file)
        writer_train.writerow(["Total Parameters", total_params])

    with open(f'./logs/{run_name}_params.csv', 'w', newline='') as file:
        writer_test = csv.writer(file)
        writer_test.writerow(["Total Parameters", total_params])
    

    # Close the TensorBoard writer
    writer.close()


    
def Test():
    # Load data from folders
    data = datasets.ImageFolder(root='./Data/Fruit', transform=transform)

    # Create a dictionary that maps indices to class names
    idx_to_class = {v: k for k, v in data.class_to_idx.items()}
    # Define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create an instance of the model
    #model = Net()
    model.load_state_dict(torch.load('./TrainedModels/fruitmodel.pt'))    # Load the trained model
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
            Train(hparams, 'fruitmodel-cnn-large', CNNFruitModelnew.Net())
            Train(hparams, 'fruitmodel-cnn-small', CNNFruitModel.Net())
            #Train(hparams, 'fruitmodel-cnn-small', CNNFruitModelSmall.Net())
            Train(hparams, 'fruitmodel-large-linear', CNNFruitModelNoCNNLarge.Net())
            Train(hparams, 'fruitmodel-small-linear', CNNFruitModelNoCNNSmall.Net())
        elif user_input == '2':
            Test()
        else:
            print('Invalid input')
    

