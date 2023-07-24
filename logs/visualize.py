import pandas as pd
import plotly.graph_objects as go
import os
import dash
import dash_core_components as dcc
import dash_html_components as html

##TODO: Add pictures that we are classifying, maybe add optimizers, add descriptions of each plot.

# Initialize Dash app
app = dash.Dash(__name__)

# Define distinct colors for each model
model_colors = {
    "fruitmodel-cnn-small": 'blue',
    "fruitmodel-cnn-large": 'green',
    "fruitmodel-large-linear": 'red',
    "fruitmodel-small-linear": 'yellow'
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

# Create three figures
fig_train = go.Figure()
fig_test = go.Figure()
fig_params = go.Figure()

# Loop over each training file
for file in train_files:
    # Read the file into a pandas DataFrame
    df = pd.read_csv(file)
    
    # Group by epoch and calculate the average training loss
    average_loss = df.groupby("Epoch")["Training Loss"].mean()
    
    # Get the model name from the filename
    model_name = file.replace("_train.csv", "")
    
    # Add a trace to the figure
    fig_train.add_trace(go.Scatter(x=average_loss.index, y=average_loss, mode='lines', name=f'{model_name} (Params: {model_params[model_name]})', line=dict(color=model_colors[model_name])))

# Update the layout of the figure
fig_train.update_layout(title='Average Training Loss per Epoch for each Model', xaxis_title='Epoch', yaxis_title='Average Training Loss',hovermode='x unified')

# Loop over each test file
for file in test_files:
    # Read the file into a pandas DataFrame
    df = pd.read_csv(file)
    
    # Get the model name from the filename
    model_name = file.replace("_test.csv", "")
    
    # Add a trace to the figure
    fig_test.add_trace(go.Scatter(x=df["Epoch"], y=df["Test Loss"], mode='lines',name=f'{model_name} (Params: {model_params[model_name]})', line=dict(color=model_colors[model_name])))

# Update the layout of the figure
fig_test.update_layout(title='Test Loss per Epoch for each Model', xaxis_title='Epoch', yaxis_title='Test Loss', hovermode='x unified')

# Loop over each model
for model, params in model_params.items():
    # Add a bar to the figure
    fig_params.add_trace(go.Bar(x=[model], y=[params], name=model, marker=dict(color=model_colors[model])))

# Update the layout of the figure
fig_params.update_layout(title='Parameter Count for each Model', xaxis_title='Model', yaxis_title='Parameter Count')

# Define the layout for your Dash app
app.layout = html.Div(children=[
    html.H1(children='NN Fruit Classifier | Convolution vs Linear'),

    html.Div(children='''
        A dashboard for visualizing training losses, test losses, and parameters count.
    ''',style={'margin-bottom': '50px','font-weight': 'bold'}),
    html.Div(children='''
    A visualization of what a typical feed-forward neural network looks like. The input layer is the image of the fruit, the hidden layers are the neurons, and the output layer is the classification.
'''),
    html.Img(src=app.get_asset_url('architecture.png'), style={'width': '100%', 'height': 'auto'}),

    html.Div(children='''
        A simple bar chart comparing the number of parameters for each model. More parameters will typically mean more memory usage, higher train times, and slower inference.
             If the architecture is designed well, more parameters will also mean better performance.
    '''),
    dcc.Graph(
        id='example-graph-params',
        figure=fig_params
    ),
    html.Div(children='''
        The train loss for each model. The train loss is the average loss for each epoch. Our models were trained a data set of 30,000
             labeled images of various different fruits and vegetables.
    '''),
    dcc.Graph(
        id='example-graph-training',
        figure=fig_train
    ),
        html.Div(children='''
        The test loss as evaluated on a real word data set of 60 images taken from the internet. As we can see, training loss doesn't not always indicate good performance on real world
                 data. This is why we use a test set to evaluate our model. The fully linear models suffer from a phenomena known as overfitting.
    '''),

    dcc.Graph(
        id='example-graph-test',
        figure=fig_test
    ),


])

# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)
