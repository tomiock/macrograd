import pytest
import numpy as np
import time
import unittest

from macrograd.tensor import Tensor


from macrograd import Tensor, e
from macrograd.tensor import _to_var
from macrograd.model import Model, Linear, SGD_Optimizer
from sklearn.datasets import make_moons
from tqdm import tqdm

from macrograd.functions import log2, relu, sigmoid


def BCE(y_pred: Tensor, y_true: Tensor) -> Tensor:
    y_true = _to_var(y_true)
    y_true = y_true.reshape(-1, 1)

    y_true.requires_grad = False

    loss_val = y_true * log2(y_pred) + (1 - y_true) * log2(1 - y_pred)

    return -1 * (loss_val.sum() / y_true.data.size)


def softmax(x: Tensor):
    e_x = e**x
    return e_x / (e_x.sum(axis=1, keepdims=True))


class NeuralNetwork(Model):
    def __init__(self, in_dims, out_dims):
        super(NeuralNetwork, self).__init__()

        self.l1 = Linear(in_dims, 24)
        self.l4 = Linear(24, out_dims)

    def __call__(self, data):
        data = self.l1(data)
        data = relu(data)

        data = self.l4(data)
        data = sigmoid(data)

        return data


class TestMacrogradModel(unittest.TestCase):
    def setUp(self):
        # --- Data Loading and Preprocessing ---
        self.X, self.y = make_moons(n_samples=100, noise=0.1, random_state=42)
        self.X = Tensor(self.X, requires_grad=False)
        self.y_one_hot = Tensor(np.eye(2)[(self.y.flatten() + 1).astype(int) // 2])
        self.lr = 0.1

        self.input_size = 2
        self.output_size = 1

        self.model = NeuralNetwork(2, 1)
        self.optimizer = SGD_Optimizer(self.lr)

        self.parameters = self.model.init_params()

        self.num_epochs = 1000  # Reduced for testing

    def test_training_loop(self):
        initial_loss = None
        final_loss = None
        losses = []

        for epoch in tqdm(range(self.num_epochs)):
            y_pred = self.model(self.X)

            loss = BCE(y_pred, self.y)
            losses.append(loss.data.item())

            if epoch == 0:
                initial_loss = loss.data.item()

            self.optimizer.step(loss, self.parameters)

            final_loss = loss.data.item()

        predictions = self.model(self.X)

        self.assertIsNotNone(initial_loss)
        self.assertIsNotNone(final_loss)
        # problem here
        self.assertGreater(initial_loss, final_loss)  # Loss should decrease
        print(initial_loss, final_loss)

        # --- Predictions --- after training
        predictions = (predictions.data > 0.5).astype(int)
        accuracy = np.mean(predictions == self.y.reshape(-1, 1))
        print(f"Accuracy: {accuracy * 100:.2f}%")
        self.assertGreaterEqual(accuracy, 0.75)  # Check for reasonable accuracy

    def test_bce_loss(self):
        y_pred = Tensor(np.array([0.8, 0.2, 0.9, 0.1]))
        y_true = Tensor(np.array([1, 0, 1, 0]))
        loss = BCE(y_pred, y_true)
        self.assertIsInstance(loss, Tensor)
        self.assertFalse(np.isnan(loss.data))

        # Test with edge cases (close to 0 and 1)
        y_pred_edge = Tensor(np.array([0.999, 0.001]))
        y_true_edge = Tensor(np.array([1, 0]))
        loss_edge = BCE(y_pred_edge, y_true_edge)
        self.assertFalse(np.isnan(loss_edge.data))


class TestPolynomialRegression(unittest.TestCase):
    # --- Shared setup method ---
    def setUp(self):
        """
        This method runs *before* each test.  It sets up the common data
        and initial parameters.
        """
        self.m = 100  # number of training points
        self.w_true = np.array([-1.0, 2.0, 1.0])  # True weights

        self.XX = 4 * np.random.rand(self.m) - 2
        self.YY = (
            self.w_true[0] + self.w_true[1] * self.XX + self.w_true[2] * (self.XX**2)
        )
        self.YY += np.random.normal(scale=0.3, size=self.YY.size)
        self.lr = 0.01
        self.epochs = 200
        self.tolerance = 0.5

        # Calculate initial loss (vectorized)
        X = np.stack([np.ones(self.m), self.XX, self.XX**2], axis=1)
        self.initial_loss = np.sum(
            (X @ np.array([0, 0, 0]).reshape(-1, 1) - self.YY.reshape(-1, 1)) ** 2
        )

    def test_convergence_vectorized(self):
        w = Tensor(np.random.normal(size=(3, 1)), requires_grad=True)  # 3x1 vector
        training_loss = []

        t = time.time()

        for _ in range(self.epochs):
            w.zero_grad()
            batch_loss = 0

            # Vectorized loss calculation
            X = Tensor(
                np.stack([np.ones(self.m), self.XX, self.XX**2], axis=1),
                requires_grad=False,
            )  # m x 3
            Y = Tensor(
                self.YY.reshape(-1, 1), requires_grad=False
            )  # Convert Y to a column vector (m x 1)

            out = X @ w  # (m x 3) @ (3 x 1) -> (m x 1)
            loss = ((out - Y) ** 2).sum()

            loss.backprop()
            batch_loss += loss.data
            training_loss.append(batch_loss)

            w = w - self.lr * w.grad / self.m  # Correct weight update

        vectorized_time = time.time() - t
        print("Vectorized Training took (secs):", vectorized_time)

        # --- Convergence Checks ---
        print(f"Initial loss: {self.initial_loss}")
        print(f"Final loss: {batch_loss}")
        self.assertLess(batch_loss, self.initial_loss * 0.1)


if __name__ == "__main__":
    unittest.main()
