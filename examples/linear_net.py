import matplotlib.pyplot as plt
import numpy as np
from macrograd import Tensor, e
from macrograd.tensor import _to_var
from sklearn.datasets import make_moons

from macrograd.functions import log2, sigmoid, relu

import warnings

np.random.seed = 42
warnings.filterwarnings("ignore")


def BCE(y_pred: Tensor, y_true: Tensor) -> Tensor:
    y_true = _to_var(y_true)
    y_true = y_true.reshape(-1, 1)

    y_true.requires_grad = False

    loss_val = y_true * log2(y_pred) + (1 - y_true) * log2(1 - y_pred)

    return -1 * (loss_val.sum() / y_true.arr.size)


# --- Data Loading and Preprocessing ---
X, y = make_moons(n_samples=100, noise=0.1, random_state=42)
X = Tensor(X, requires_grad=False)
y_one_hot = Tensor(np.eye(2)[(y.flatten() + 1).astype(int) // 2])


input_size = 2
hidden_size = 16
output_size = 1

w_1 = Tensor(np.random.randn(input_size, hidden_size), requires_grad=True)
b_1 = Tensor(np.zeros((1, hidden_size)), requires_grad=True)
w_2 = Tensor(np.random.randn(hidden_size, output_size), requires_grad=True)
b_2 = Tensor(np.zeros((1, output_size)), requires_grad=True)

parameters = [w_1, b_1, w_2, b_2]

# --- Training Loop ---
lr = 0.1
num_epochs = 1000


def model(input):
    linear_1 = (input @ w_1) + b_1
    z_1 = relu(linear_1)

    linear_2 = (z_1 @ w_2) + b_2
    z_2 = sigmoid(linear_2)

    return z_2


losses = []
for epoch in range(num_epochs):
    for param in parameters:
        param.zeroGrad()

    y_pred = model(X)

    loss = BCE(y_pred, y)
    loss.backprop()

    w_1 = w_1 - w_1.grad * lr
    w_2 = w_2 - w_2.grad * lr

    b_1 = b_1 - b_1.grad * lr
    b_2 = b_2 - b_2.grad * lr

    losses.append(loss.arr.item())

# Create a grid of points to evaluate the model
h = 0.05  # Step size in the mesh
x_min, x_max = X.arr[:, 0].min() - 1, X.arr[:, 0].max() + 1
y_min, y_max = X.arr[:, 1].min() - 1, X.arr[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Predict the function value for the whole grid
grid_points = Tensor(np.c_[xx.ravel(), yy.ravel()])
print(grid_points.shape)
Z = model(grid_points)
Z = Z.arr.reshape(xx.shape)  # Reshape back to the grid shape

# Plot the contour and training examples
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)  # Decision boundary
plt.scatter(
    X.arr[:, 0], X.arr[:, 1], c=y.flatten(), s=40, cmap=plt.cm.Spectral
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
predictions = (predictions.arr > 0.5).astype(
    int
)  # Threshold at 0.5 for binary classification
accuracy = np.mean(predictions == y.reshape(-1, 1))  # calculate the accuracy
print(f"Accuracy: {accuracy * 100:.2f}%")
