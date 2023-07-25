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
        "color": 'yellow',
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

    # Add a trace to the figure
    fig_train.add_trace(go.Scatter(x=average_loss.index, y=average_loss, mode='lines', name=f'{model_name} (Params: {model_params})', line=dict(color=model_color)))

# Update the layout of the figure
fig_train.update_layout(title='Average Training Loss per Epoch for each Model (lower is better)', xaxis_title='Epoch', yaxis_title='Average Training Loss',hovermode='x unified')

# Loop over each test file
for file in test_files:
    # Read the file into a pandas DataFrame
    df = pd.read_csv(file)

    # Get the model info from the filename
    file_base = file.replace("_test.csv", "")
    model_name = model_info[file_base]["name"]
    model_color = model_info[file_base]["color"]
    model_params = model_info[file_base]["params"]

    # Add a trace to the figure
    fig_test.add_trace(go.Scatter(x=df["Epoch"], y=df["Test Loss"], mode='lines',name=f'{model_name} (Params: {model_params})', line=dict(color=model_color)))

# Update the layout of the figure
fig_test.update_layout(title='Test Loss per Epoch for each Model (lower is better)', xaxis_title='Epoch', yaxis_title='Test Loss', hovermode='x unified')

# Loop over each model
for file_base, info in model_info.items():
    # Add a bar to the figure
    fig_params.add_trace(go.Bar(x=[info["name"]], y=[info["params"]], name=info["name"], marker=dict(color=info["color"])))

# Update the layout of the figure
fig_params.update_layout(title='Parameter Count for each Model', xaxis_title='Model', yaxis_title='Parameter Count')

# Define the layout for your Dash app
app.layout = html.Div(children=[
    html.H1(children='NN Fruit Classifier | With Convolution vs Without Convolution'),

    html.Div(children='''
        A dashboard for visualizing training losses, test losses, and parameters count.
    ''',style={'margin-bottom': '50px','font-weight': 'bold'}),
    html.Img(src=app.get_asset_url('architecture.png'), style={'width': '100%', 'height': 'auto'}),
    html.H2(children='What is Convolution?'),
    html.Div(children='''
    Convolution is a technique for learning patterns in structured data. It can be thought of as passing a filter over an input data and learning
    the best way to identify certain aspects of its inputs. It is best used when structure of the data is important.
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
        html.Div(children='''
        The test loss as evaluated on a real word data set of 60 images taken from the internet. As we can see, training loss doesn't not always indicate good performance on real world
                 data. This is why we use a test set to evaluate our model. The non-convolutional models suffer from a phenomena known as overfitting.
    ''',style={'font-weight': 'bold','font-size': '15px'}),

    dcc.Graph(
        id='example-graph-test',
        figure=fig_test
    ),
    html.Div(children='''
        The smaller model won! It performed the best because it did not suffer from overfitting, and used convolution to accurately learn training images.
    ''',style={'font-weight': 'bold','font-size': '30px'}),
    
    dcc.Graph(
    id='example-graph-params2',
    figure=fig_params
    ),

])

# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)

