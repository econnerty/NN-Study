import pandas as pd
import plotly.graph_objects as go
import os

# Define distinct colors for each model
model_colors = {
    "fruitmodel-cnn-small": 'blue',
    "fruitmodel-cnn-large": 'green',
    "fruitmodel-linear-large": 'red',
    "fruitmodel-linear-small": 'yellow'
}

# Define the total parameters for each model
model_params = {
    "fruitmodel-cnn-small": pd.read_csv("fruitmodel-cnn-small_params.csv", header=None).iloc[0, 1], #param is located in the second column and first row
    "fruitmodel-cnn-large": pd.read_csv("fruitmodel-cnn-large_params.csv", header=None).iloc[0, 1],
    "fruitmodel-large-linear": pd.read_csv("fruitmodel-large-linear_params.csv", header=None).iloc[0, 1],
    "fruitmodel-small-linear": pd.read_csv("fruitmodel-small-linear_params.csv", header=None).iloc[0, 1],
}

# Get the list of all csv files in the current directory
all_files = [file for file in os.listdir() if file.endswith('.csv')]

# Separate the training and test files
train_files = [file for file in all_files if "train" in file]
test_files = [file for file in all_files if "test" in file]

# Create a figure
fig = go.Figure()

# Loop over each training file
for file in train_files:
    # Read the file into a pandas DataFrame
    df = pd.read_csv(file)
    
    # Group by epoch and calculate the average training loss
    average_loss = df.groupby("Epoch")["Training Loss"].mean()
    
    # Get the model name from the filename
    model_name = file.replace("_train.csv", "")
    
    # Add a trace to the figure
    fig.add_trace(go.Scatter(x=average_loss.index, y=average_loss, mode='lines', name=f'{model_name} (Params: {model_params[model_name]})', line=dict(color=model_colors[model_name])))

# Update the layout of the figure
fig.update_layout(title='Average Training Loss per Epoch for each Model', xaxis_title='Epoch', yaxis_title='Average Training Loss')

# Display the figure
fig.show()

# Create a figure
fig = go.Figure()

# Loop over each test file
for file in test_files:
    # Read the file into a pandas DataFrame
    df = pd.read_csv(file)
    
    # Get the model name from the filename
    model_name = file.replace("_test.csv", "")
    
    # Add a trace to the figure
    fig.add_trace(go.Scatter(x=df["Epoch"], y=df["Test Loss"], mode='lines',name=f'{model_name} (Params: {model_params[model_name]})', line=dict(color=model_colors[model_name])))

# Update the layout of the figure
fig.update_layout(title='Test Loss per Epoch for each Model', xaxis_title='Epoch', yaxis_title='Test Loss')

# Display the figure
fig.show()

# Create a new figure for the parameter counts
fig = go.Figure()

# Loop over each model
for model, params in model_params.items():
    # Add a bar to the figure
    fig.add_trace(go.Bar(x=[model], y=[params], name=model, marker=dict(color=model_colors[model])))

# Update the layout of the figure
fig.update_layout(title='Parameter Count for each Model', xaxis_title='Model', yaxis_title='Parameter Count')

# Display the figure
fig.show()
