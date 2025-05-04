from math import prod

import matplotlib.pyplot as plt
import numpy as np

from macrograd import Tensor
from macrograd.engine import get_default_graph
from macrograd.tensor import relu
from sklearn.datasets import make_moons

from macrograd.model import SGD_MomentumOptimizer, Model, Linear

np.random.seed = 42


def BCE(y_pred: Tensor, y_true) -> Tensor:
    if not isinstance(y_true, Tensor):
        y_true = y_true.reshape((-1, 1))
        y_true = Tensor(array=y_true)

    y_true.requires_grad = False

    loss_val = y_true * y_pred.log(base=2) + (1 - y_true) * 1 - y_pred.log(base=2)
    size_y_true = prod(y_true.node.shape)

    return -1 * (loss_val.sum() / size_y_true)


graph = get_default_graph()

# --- Data Loading and Preprocessing ---
X, y = make_moons(n_samples=100, noise=0.2, random_state=42)
y = np.array(y)
X = Tensor(X, requires_grad=False)
y_one_hot = Tensor(np.eye(2)[(y.flatten() + 1).astype(int) // 2])


def sigmoid(x: Tensor):
    return x / (1 + (-x).exp())


class NeuralNetwork(Model):
    def __init__(self, in_dims, out_dims):
        super(NeuralNetwork, self).__init__()

        self.l1 = Linear(in_dims, 24)
        self.l2 = Linear(24, out_dims)

    def __call__(self, data):
        data = self.l1(data)
        data = relu(data)

        data = self.l2(data)
        data = sigmoid(data)

        return data


model = NeuralNetwork(2, 1)
model.init_params()
parameters = model.parameters

num_epochs = 2000
losses = []

optimizer = SGD_MomentumOptimizer(learning_rate=0.01, alpha=0.9, params_copy=parameters)

for epoch in range(num_epochs):
    # forward pass
    y_pred = model(X)

    # loss calculation
    loss = BCE(y_pred, y)

    graph.visualize()

    loss.realize()

    # parameter update
    parameters = optimizer.step(loss, parameters)

    losses.append(loss.data.item())

"""
# Create a grid of points to evaluate the model
h = 0.05  # Step size in the mesh
x_min, x_max = X.data[:, 0].min() - 1, X.data[:, 0].max() + 1
y_min, y_max = X.data[:, 1].min() - 1, X.data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Predict the function value for the whole grid
grid_points = Tensor(np.c_[xx.ravel(), yy.ravel()])
Z = model(grid_points)
Z = Z.data.reshape(xx.shape)  # Reshape back to the grid shape

# Plot the contour and training examples
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)  # Decision boundary
plt.scatter(
    X.data[:, 0], X.data[:, 1], c=y.flatten(), s=40, cmap=plt.cm.Spectral
)  # Data points
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("Decision Boundary and Data Points")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()


# --- Loss Plot ---
plt.figure(figsize=(8, 6))
plt.plot(losses)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.yscale("log")
plt.show()

# --- Predictions ---
predictions = model(X)
predictions = (predictions.data > 0.5).astype(
    int
)  # Threshold at 0.5 for binary classification
accuracy = np.mean(predictions == y.reshape(-1, 1))  # calculate the accuracy
print(f"Accuracy: {accuracy * 100:.2f}%")
"""
