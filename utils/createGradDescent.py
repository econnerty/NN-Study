# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the function and its gradient
def f(x, y):
    return x**2 + y**2

def grad_f(x, y):
    return np.array([2*x, 2*y])

# Generate x and y values
x = np.linspace(-10, 10, 100)
y = np.linspace(-10, 10, 100)
x, y = np.meshgrid(x, y)

# Compute z values
z = f(x, y)

# Create a 3D plot
fig = plt.figure(figsize=(12, 5))

ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(x, y, z, cmap='viridis')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('f(X, Y)')
ax1.set_title('3D Plot')

ax2 = fig.add_subplot(122)
contour = ax2.contour(x, y, z, levels=np.logspace(0, 3, 10))
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_title('Contour Plot')
plt.show()

# Set the learning rate and number of iterations
alpha = 0.05
n_iterations = 50

# Initialize the starting point
point = np.array([5, -7])
path = [point]

# Run gradient descent
for _ in range(n_iterations):
    grad = grad_f(*point)
    point = point - alpha * grad
    path.append(point)

# Convert the path to a numpy array for easier indexing
path = np.array(path)

# Plot the 3D surface with the path of gradient descent
fig = plt.figure(figsize=(12, 5))

ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(x, y, z, cmap='viridis', alpha=0.5)
ax1.plot3D(path[:,0], path[:,1], f(path[:,0], path[:,1]), 'r-')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('f(X, Y)')
ax1.set_title('3D Plot')

ax2 = fig.add_subplot(122)
contour = ax2.contour(x, y, z, levels=np.logspace(0, 3, 10))

# Plot the path with arrows indicating the gradient direction
for i in range(0, len(path) - 1, 2):
    grad = grad_f(*path[i])
    ax2.quiver(path[i,0], path[i,1], -alpha * grad[0], -alpha * grad[1], color='b', angles='xy', scale_units='xy', scale=.5)

ax2.plot(path[:,0], path[:,1], 'r-')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_title('Contour Plot')
plt.show()
