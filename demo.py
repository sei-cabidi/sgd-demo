import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import argparse

torch.manual_seed(29)

# ----------------------------- #
# Plotting Utils                #
# ----------------------------- #
class ScatterPlot:
    def __init__(self, color='red'):
        self.fig, self.ax = plt.subplots()
        plt.grid(True)
        plt.axhline(0)
        plt.axvline(0)
        self.ax.set_xlim(-10, 10)
        self.ax.set_ylim(-10, 10)
        self.points = []
        self.color = color
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self._onclick)

    def add_points(self, points, color='red'):
        for x, y in points:
            self.ax.scatter(x, y, color=color)

    def _onclick(self, event):
        if event.inaxes != self.ax:
            return
        x = event.xdata
        y = event.ydata
        self.points.append((x, y))
        self.ax.scatter(x, y, color=self.color)
        self.fig.canvas.draw()

    def show(self):
        plt.show()

def plot_weights_biases(model, X, Y):
    # Get the weights and biases from the model
    weight = model.linear.weight.detach().numpy()
    bias = model.linear.bias.detach().numpy()

    # Generate x-values for the line
    x = np.linspace(-10, 10, 100)

    # Calculate y-values using the weights and biases
    y = (weight * x + bias).squeeze()

    # Plot error bars
    for i, (x_i, y_i) in enumerate(zip(X, Y)):
        y_hat = model(x_i).detach().cpu()
        x_values = [float(x_i), float(x_i)]
        y_values = [float(y_i), float(y_hat)]
        plt.plot(x_values, y_values, 'bo', linestyle="--")
    
    # Plot the line
    plt.grid(True)
    plt.axhline(0)
    plt.axvline(0)
    plt.plot(x, y, color='green', label='Linear Model: y = w * x + b')
    plt.xlabel('Input 1')
    plt.ylabel('Input 2')
    plt.title('Online Linear Regression')
    plt.legend()
    plt.grid(True)
    plt.show()

# This takes care of command line argument parsing for you!
# To access a specific argument, simply access args.<argument name>.
# For example, to get the learning rate, you can use `args.learning_rate`.
parser = argparse.ArgumentParser()
parser.add_argument("plot", type=int, default=0, help='whether to plot')
parser.add_argument("gather", type=int, default=0, help='whether to gather points')
parser.add_argument("n_samples", type=int, default=0, 
                    help='number of samples to collect')
parser.add_argument("learning_rate", type=float, default=0.00001,
                    help='learning rate for stochastic gradient descent')
args = parser.parse_args()
args.plot = bool(args.plot)
args.gather = bool(args.gather)
print(args)
    
# ----------------------------- #
# Gather Points                 #
# ----------------------------- #
if args.gather:
    scatter_plot = ScatterPlot(color='red')
    scatter_plot.show()
    points_a = scatter_plot.points
    plt.ioff()
    print("Coordinates recorded")
else:
    # Fake equation: y = 3 * x + noise
    x = np.linspace(-10, 10, args.n_samples)
    y = 3 * x + np.random.normal(0.0, 2.0, size=x.shape[0])
    points_a = np.vstack((x,y)).T

    if args.plot:
        scatter_plot = ScatterPlot(color='red')
        scatter_plot.add_points(points_a)
        scatter_plot.show()

# ----------------------------- #
# Form Dataset                  #
# ----------------------------- #
data = torch.Tensor(points_a)
X = data[:,0].unsqueeze(1)
Y = data[:,1].unsqueeze(1)
print("Dataset has been formed")
print(f"X: {X.shape}\nY: {Y.shape}")

# ----------------------------- #
# Create Neural Network         #
# ----------------------------- #

# Define the one-layer neural network
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

        # 1 input feature, 1 output feature
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

# Create an instance of the neural network
model = NeuralNetwork()

# Define the loss function
# Mean Squared Error 
# (predicted y value - true y value)^2 
criterion = nn.MSELoss()

# Define the optimizer: Stochastic Gradient Descent
optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)

# Turn on interactive plotting
if args.plot:
    plt.ion()

# Train the model
num_epochs = 5000
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X)
    
    # Measure the error between predicted y valued and true y values
    loss = criterion(outputs, Y)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Print the loss for every 100 epochs
    if (epoch + 1) % 50 == 0:
        print(f'Epoch: {epoch+1}, Loss: {loss.item()}')

        if args.plot:
            # Plot the points
            plt.pause(0.5)
            plt.cla()
            plt.scatter(data[:,0], data[:,1], color='red')

            # Plot the learned line
            plot_weights_biases(model, X, Y)