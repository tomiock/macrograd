import numpy as np
import matplotlib.pyplot as plt
import time

from macrograd import Tensor

m = 100  # number of training points
w_true = [-1.0, 2.0, 1.0]  # True weights

XX = 4 * np.random.rand(m) - 2
#XX = np.array([1, 2])
YY = w_true[0] + w_true[1] * XX + w_true[2] * (XX**2)
#YY += np.random.normal(scale=0.3, size=YY.size)

# Initialize weights
w = Tensor(np.random.normal(size=(3, 1)), requires_grad=True)  # 3x1 vector
#w = Tensor([[1.], [2.], [1.]], requires_grad=True)  # 3x1 vector


lr = 0.01
epochs = 1000
training_loss = []
batch_loss = 0

t = time.time()

for _ in range(epochs):
    w.zero_grad()

    batch_loss = 0
    X = Tensor(np.stack([np.ones(m), XX, XX**2], axis=1), requires_grad=False)
    y = Tensor(YY.reshape(-1, 1), requires_grad=False)

    out = X @ w
    loss = ((out - y) ** 2).sum()

    #visualize_graph(loss, filename='poly_reg')

    loss.backprop()  # Use backprop
    batch_loss += loss.arr

    training_loss.append(batch_loss)

    w = w - lr * w.grad / m

print("Training took (secs):", time.time() - t)

X = np.stack([np.ones(m), XX, XX**2], axis=1)
initial_loss = np.sum((X @ np.array([0,0,0]).reshape(-1,1) - YY.reshape(-1,1))**2)

print(f"Initial loss: {initial_loss}")
print(f"Final loss: {batch_loss}")

# Check if the estimated weights are close to the true weights.
tolerance = 0.5  # Adjust as needed
print(f"Estimated w: {w.arr.flatten()}, True w: {w_true}")

plt.title('Our estimated fit')
plt.plot(XX, YY, '.')
x_vals = np.arange(-2, 2, 0.1)
y_vals = w.arr[0,0] + w.arr[1,0] * x_vals + w.arr[2,0] * x_vals**2  # Use .arr to access NumPy arrays
plt.plot(x_vals, y_vals, 'r')
plt.show()

plt.plot(training_loss)
plt.show()
