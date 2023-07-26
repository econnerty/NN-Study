import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# Define the sizes of each layer in the network
layer_sizes = [20, 50, 50, 30]

# Create a new figure
fig, ax = plt.subplots(figsize=(10, 6))

# Define the vertical positions of each layer
layer_top = 1.0
layer_bottom = 0.0

max_layer_size = max(layer_sizes)

# Declare neuron_positions_map
neuron_positions_map = {}

for n, layer_size in enumerate(layer_sizes):
    # Compute the horizontal positions of each neuron
    layer_left = n / (len(layer_sizes) - 1)
    
    # Create an adjustment factor for the y-position to center the neurons
    y_adjustment = (max_layer_size - layer_size) / (2*max_layer_size)

    # Draw the neurons
    neuron_positions = []
    for m in range(layer_size):
        y_position = (m / max_layer_size) + y_adjustment
        circle = plt.Circle((layer_left, y_position), 0.01, color='blue', ec='black', zorder=4)
        ax.add_artist(circle)
        neuron_positions.append(y_position)
        
        # Add outer ring to denote bias
        if n > 0:
            outer_circle = plt.Circle((layer_left, y_position), 0.015, fill=False, ec='red', lw=0.5)
            ax.add_artist(outer_circle)
    
    # Draw the weights
    if n > 0:
        prev_neuron_positions = neuron_positions_map[n-1]
        for o, o_pos in enumerate(prev_neuron_positions):
            for p, p_pos in enumerate(neuron_positions):
                line = plt.Line2D((layer_left - 1 / (len(layer_sizes) - 1), layer_left), 
                                  (o_pos, p_pos), 
                                  color='gray', linewidth=0.2)
                ax.add_artist(line)
    neuron_positions_map[n] = neuron_positions

# Add image to the left of the input layer
img = mpimg.imread('banana.jpg')  # replace 'image_path.jpg' with your image file path
imgbox = OffsetImage(img, zoom=0.1)  # adjust zoom level to suit your needs
ab = AnnotationBbox(imgbox, (-0.2, 0.5))
ax.add_artist(ab)

# Add arrow from image to input layer
plt.annotate('', xy=(0, 0.5), xytext=(-0.1, 0.5), 
             arrowprops=dict(facecolor='black', shrink=0.5))

# Add word to the right of the output layer
plt.text(1.15, 0.5, 'banana', va='center')

# Add arrow from output layer to word
plt.annotate('', xy=(1.1, 0.5), xytext=(1.05, 0.5), 
             arrowprops=dict(facecolor='black', shrink=0.05))

plt.text(0.0, 1.1, 'Input layer', ha='center')
plt.text(1.0, 1.1, 'Output layer', ha='center')
plt.title('Neural network architecture')

# Legend

input_legend = plt.Line2D((0,1),(0,0), color='b', marker='o', linestyle='')
weight_legend = plt.Line2D((0,1),(0,0), color='gray')
bias_legend = plt.Line2D((0,1),(0,0), marker='o', color='w', markerfacecolor='b', markersize=5, markeredgewidth=0.5, markeredgecolor='r')
#plt.legend([input_legend, weight_legend, bias_legend], ['Neurons', 'Weights', 'Biases'])

ax.set_xlim(-0.2, 1.2)
ax.set_ylim(-0.1, 1.2)
plt.axis('off')
plt.show()
