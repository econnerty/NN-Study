import pandas as pd
import plotly.graph_objects as go
import os
import dash
import dash_core_components as dcc
import dash_html_components as html

# Initialize Dash app
app = dash.Dash(__name__)

# Define distinct colors for each model
model_info = {
    "fruitmodel-cnn-small": {
        "name": "With Convolution: Small",
        "color": 'blue',
        "params": pd.read_csv("fruitmodel-cnn-small_params.csv", header=None).iloc[0, 1]
    },
    "fruitmodel-cnn-large": {
        "name": "With Convolution: Large",
        "color": 'green',
        "params": pd.read_csv("fruitmodel-cnn-large_params.csv", header=None).iloc[0, 1]
    },
    "fruitmodel-small-linear": {
        "name": "No Convolution: Small",
        "color": 'red',
        "params": pd.read_csv("fruitmodel-small-linear_params.csv", header=None).iloc[0, 1]
    },
    "fruitmodel-large-linear": {
        "name": "No Convolution: Large",
        "color": 'orange',
        "params": pd.read_csv("fruitmodel-large-linear_params.csv", header=None).iloc[0, 1]
    }
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

    # Get the model info from the filename
    file_base = file.replace("_train.csv", "")
    model_name = model_info[file_base]["name"]
    model_color = model_info[file_base]["color"]
    model_params = model_info[file_base]["params"]

    if model_name in ['With Convolution: Small', 'With Convolution: Large']:
        # If the model is one of the ones to highlight, add a 'border' line
        fig_train.add_trace(go.Scatter(
            x=average_loss.index,
            y=average_loss,
            mode='lines',
            line=dict(color='black', width=5), # Set color and width of the 'border'
            showlegend=False, # We don't want this trace to show up in the legend
            hoverinfo='none' # Prevent hover text from showing up for this trace

        ))

    # Add the original line (which will be overlaid on the 'border' line if it exists)
    fig_train.add_trace(go.Scatter(
        x=average_loss.index,
        y=average_loss,
        mode='lines',
        name=f'{model_name} (Params: {model_params})',
        line=dict(color=model_color, width=2) # You may need to adjust the width depending on the 'border' width
    ))


# Update the layout of the figure
fig_train.update_layout(title='Average Training Loss per Epoch for each Model (lower is better)', xaxis_title='Epoch', yaxis_title="Training Loss (% of 100)",hovermode='x unified')

# Loop over each test file
for file in test_files:
    # Read the file into a pandas DataFrame
    df = pd.read_csv(file)

    # Get the model info from the filename
    file_base = file.replace("_test.csv", "")
    model_name = model_info[file_base]["name"]
    model_color = model_info[file_base]["color"]
    model_params = model_info[file_base]["params"]

    if model_name in ['With Convolution: Small', 'With Convolution: Large']:
        # If the model is one of the ones to highlight, add a 'border' line
        fig_test.add_trace(go.Scatter(
            x=df["Epoch"],
            y=df["Test Loss"],
            mode='lines',
            line=dict(color='black', width=5), # Set color and width of the 'border'
            showlegend=False, # We don't want this trace to show up in the legend
            hoverinfo='none' # Prevent hover text from showing up for this trace
        ))

    # Add the original line (which will be overlaid on the 'border' line if it exists)
    fig_test.add_trace(go.Scatter(
        x=df["Epoch"],
        y=df["Test Loss"],
        mode='lines',
        name=f'{model_name} (Params: {model_params})',
        line=dict(color=model_color, width=2) # You may need to adjust the width depending on the 'border' width
    ))

# Update the layout of the figure
fig_test.update_layout(title='Test Loss per Epoch for each Model (lower is better)', xaxis_title='Epoch', yaxis_title="Test Loss (% of 100)", hovermode='x unified')

# Loop over each model
for file_base, info in model_info.items():
    if info["name"] in ['With Convolution: Small', 'With Convolution: Large']:
        # If the model is one of the ones to highlight, give it a thick, dark border
        line = dict(color='black', width=5)
    else:
        # Otherwise, give it no border
        line = dict(width=0)
    
    # Add a bar to the figure
    fig_params.add_trace(go.Bar(
        x=[info["name"]], 
        y=[info["params"]], 
        name=info["name"], 
        marker=dict(
            color=info["color"],
            line=line
        )
    ))

# Update the layout of the figure
fig_params.update_layout(title='Parameter Count for each Model', xaxis_title='Model', yaxis_title='Parameter Count (millions)')

# Define the layout for your Dash app
app.layout = html.Div(children=[
    html.H1(children='NN Fruit Classifier | With Convolution vs Without Convolution'),

    html.Div(children='''
        A dashboard for visualizing training loss, test loss, and parameter count.
    ''',style={'margin-bottom': '50px','font-weight': 'bold'}),
    html.Img(src=app.get_asset_url('architecture.svg'), style={'width': '100%', 'height': 'auto'}),
    html.H2(children='What is Convolution?'),
    html.Div(children='''
    Convolution is a technique for learning patterns in structured data. For the purpose of image recongnition, it can be thought of as passing several filters over an input image and learning
    its most important features. It can be used anytime the input data needs a sense of spatial importance (I.E the data needs spatial information to be preserved).
''',style={'margin-bottom': '50px','font-weight': 'bold'}),
    html.Div(children='''
    A visualization of what a typical feed-forward neural network looks like. The input layer is the image of the fruit, the hidden layers are the neurons, and the output layer is the classification.
''',style={'font-weight': 'bold','font-size': '15px'}),


    html.Div(children='''
        A simple bar chart comparing the number of parameters for each model. More parameters will typically mean more memory usage, higher train times, and slower inference.
             If the architecture is designed well, more parameters will also mean better performance.
    ''',style={'font-weight': 'bold','font-size': '15px'}),
    dcc.Graph(
        id='example-graph-params',
        figure=fig_params
    ),
    html.Div(children='''
        The train loss for each model. The train loss is the average loss for each epoch. Our models were trained a data set of 30,000
             labeled images of various different fruits and vegetables.
    ''',style={'font-weight': 'bold','font-size': '15px'}),
    dcc.Graph(
        id='example-graph-training',
        figure=fig_train
    ),
        html.P('Note: Thicker lines are models that contain convolutional layers',style={'margin-bottom': '50px'}),
        html.Div(children='''
        The test loss as evaluated on a real word data set of 60 images taken from the internet. As we can see, training loss doesn't not always indicate good performance on real world
                 data. This is why we use a test set to evaluate our model. The non-convolutional models suffer from a phenomena known as overfitting.
    ''',style={'font-weight': 'bold','font-size': '15px'}),

    dcc.Graph(
        id='example-graph-test',
        figure=fig_test
    ),
    html.P('Note: Thicker lines are models that contain convolutional layers',style={'margin-bottom': '50px'}),

    html.Div(children='''
        The smaller model won! It performed the best because it did not overfit the data and used convolution to learn patterns in the images more efficiently.
    ''',style={'font-weight': 'bold','font-size': '30px'}),
    
    dcc.Graph(
    id='example-graph-params2',
    figure=fig_params
    ),

])

# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)

