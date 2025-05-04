import numpy as np
import matplotlib.pyplot as plt
import time

from tqdm import tqdm

from macrograd import Tensor
from macrograd.engine import Graph, get_default_graph

m = 100  # number of training points
w_true = [-1.0, 2.0, 1.0]  # True weights

XX = 4 * np.random.rand(m) - 2
YY = w_true[0] + w_true[1] * XX + w_true[2] * (XX**2)

# Initialize weights
w_array = np.random.normal(size=(3, 1))

lr = 0.01
epochs = 100
training_loss = []
batch_loss = 0

t = time.time()

for _ in tqdm(range(epochs)):
    g = Graph()
    X = Tensor(np.stack([np.ones(m), XX, XX**2], axis=1), requires_grad=False, graph=g)
    y = Tensor(YY.reshape(-1, 1), requires_grad=False, graph=g)
    w = Tensor(w_array, requires_grad=True)
    print(w.shape)

    batch_loss = 0

    out = X @ w

    loss = ((out - y) ** 2).sum()

    print(loss.data)
    loss.realize()

    loss.backprop()  # Use backprop


print(w)
print(g)

print("Training took (secs):", time.time() - t)

X = np.stack([np.ones(m), XX, XX**2], axis=1)
initial_loss = np.sum((X @ np.array([0,0,0]).reshape(-1,1) - YY.reshape(-1,1))**2)

print(f"Initial loss: {initial_loss}")
print(f"Final loss: {batch_loss}")

# Check if the estimated weights are close to the true weights.
tolerance = 0.5  # Adjust as needed
print(f"Estimated w: {w.data.flatten()}, True w: {w_true}")

plt.title('Our estimated fit')
plt.plot(XX, YY, '.')
x_vals = np.arange(-2, 2, 0.1)
y_vals = w.data[0,0] + w.data[1,0] * x_vals + w.data[2,0] * x_vals**2  # Use .arr to access NumPy arrays
plt.plot(x_vals, y_vals, 'r')
plt.show()

plt.plot(training_loss)
plt.show()
