import matplotlib.pyplot as plt

# Define the sizes of each layer in the network
layer_sizes = [2, 4, 2]

# Create a new figure
fig, ax = plt.subplots(figsize=(9, 6))

# Define the vertical positions of each layer
layer_top = 1.0
layer_bottom = 0.0

for n, layer_size in enumerate(layer_sizes):
    # Compute the horizontal positions of each neuron
    layer_left = n / (len(layer_sizes) - 1)
    
    # Draw the neurons
    for m in range(layer_size):
        circle = plt.Circle((layer_left, (m + 0.5) / layer_size), 0.05, color='blue', ec='black', zorder=4)
        ax.add_artist(circle)
        
        # Add outer ring to denote bias
        if n > 0:
            outer_circle = plt.Circle((layer_left, (m + 0.5) / layer_size), 0.075, fill=False, ec='red', lw=1.5)
            ax.add_artist(outer_circle)
    
    # Draw the weights
    if n > 0:
        for o in range(layer_sizes[n - 1]):
            for p in range(layer_size):
                line = plt.Line2D((layer_left - 1 / (len(layer_sizes) - 1), layer_left), ((o + 0.5) / layer_sizes[n - 1], (p + 0.5) / layer_size), color='gray')
                ax.add_artist(line)

plt.text(0.0, layer_top + 0.1, 'Input layer', ha='center')
plt.text(1.0, layer_top + 0.1, 'Output layer', ha='center')
plt.title('Neural network architecture')

# Legend
input_legend = plt.Line2D((0,1),(0,0), color='b', marker='o', linestyle='')
weight_legend = plt.Line2D((0,1),(0,0), color='gray')
bias_legend = plt.Line2D((0,1),(0,0), marker='o', color='w', markerfacecolor='b', markersize=10, markeredgewidth=1.5, markeredgecolor='r')
plt.legend([input_legend, weight_legend, bias_legend], ['Neurons', 'Weights', 'Biases'])

ax.set_xlim(-0.1, 1.1)
ax.set_ylim(layer_bottom - 0.1, layer_top + 0.2)
plt.axis('off')
plt.show()
