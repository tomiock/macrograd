import pytest
import numpy as np
import time
import unittest

from macrograd.tensor import Tensor


from macrograd import Tensor, e
from macrograd.tensor import _to_var
from sklearn.datasets import make_moons
from tqdm import tqdm

from macrograd.functions import log2, relu, sigmoid


def BCE(y_pred: Tensor, y_true: Tensor) -> Tensor:
    y_true = _to_var(y_true)
    y_true = y_true.reshape(-1, 1)

    y_true.requires_grad = False

    loss_val = y_true * log2(y_pred) + (1 - y_true) * log2(1 - y_pred)

    return -1 * (loss_val.sum() / y_true.arr.size)


def model(input, w_1, b_1, w_2, b_2):
    linear_1 = (input @ w_1) + b_1
    z_1 = relu(linear_1)

    linear_2 = (z_1 @ w_2) + b_2
    z_2 = sigmoid(linear_2)

    return z_2


class TestMacrogradModel(unittest.TestCase):
    def setUp(self):
        # --- Data Loading and Preprocessing ---
        self.X, self.y = make_moons(n_samples=100, noise=0.1, random_state=42)
        self.X = Tensor(self.X, requires_grad=False)
        self.y_one_hot = Tensor(np.eye(2)[(self.y.flatten() + 1).astype(int) // 2])

        self.input_size = 2
        self.hidden_size = 100
        self.output_size = 1

        self.w_1 = Tensor(
            np.random.randn(self.input_size, self.hidden_size), requires_grad=True
        )  # Correct shape
        self.b_1 = Tensor(
            np.zeros((1, self.hidden_size)), requires_grad=True
        )  # Correct shape, initialize to zero
        self.w_2 = Tensor(
            np.random.randn(self.hidden_size, self.output_size), requires_grad=True
        )  # Correct shape
        self.b_2 = Tensor(
            np.zeros((1, self.output_size)), requires_grad=True
        )  # Correct shape, initialize to zero

        self.parameters = [self.w_1, self.b_1, self.w_2, self.b_2]
        self.lr = 0.1
        self.num_epochs = 100  # Reduced for testing

    def test_training_loop(self):
        initial_loss = None
        final_loss = None
        losses = []

        for epoch in tqdm(range(self.num_epochs)):
            for param in self.parameters:
                param.zeroGrad()

            y_pred = model(self.X, self.w_1, self.b_1, self.w_2, self.b_2)

            loss = BCE(y_pred, self.y)
            losses.append(loss.arr.item())

            if epoch == 0:
                initial_loss = loss.arr.item()

            loss.backprop()

            self.w_1 = self.w_1 - self.w_1.grad * self.lr
            self.w_2 = self.w_2 - self.w_2.grad * self.lr
            self.b_1 = self.b_1 - self.b_1.grad * self.lr
            self.b_2 = self.b_2 - self.b_2.grad * self.lr
            final_loss = loss.arr.item()

        self.assertIsNotNone(initial_loss)
        self.assertIsNotNone(final_loss)
        self.assertGreater(initial_loss, final_loss)  # Loss should decrease

        # --- Predictions --- after training
        predictions = model(self.X, self.w_1, self.b_1, self.w_2, self.b_2)
        predictions = (predictions.arr > 0.5).astype(int)
        accuracy = np.mean(predictions == self.y.reshape(-1, 1))
        print(f"Accuracy: {accuracy * 100:.2f}%")
        self.assertGreaterEqual(accuracy, 0.75)  # Check for reasonable accuracy

        # Check for NaNs in weights and biases
        self.assertFalse(np.isnan(self.w_1.arr).any())
        self.assertFalse(np.isnan(self.b_1.arr).any())
        self.assertFalse(np.isnan(self.w_2.arr).any())
        self.assertFalse(np.isnan(self.b_2.arr).any())

    def test_bce_loss(self):
        y_pred = Tensor(np.array([0.8, 0.2, 0.9, 0.1]))
        y_true = Tensor(np.array([1, 0, 1, 0]))
        loss = BCE(y_pred, y_true)
        self.assertIsInstance(loss, Tensor)
        self.assertFalse(np.isnan(loss.arr))

        # Test with edge cases (close to 0 and 1)
        y_pred_edge = Tensor(np.array([0.999, 0.001]))
        y_true_edge = Tensor(np.array([1, 0]))
        loss_edge = BCE(y_pred_edge, y_true_edge)
        self.assertFalse(np.isnan(loss_edge.arr))

    def test_model_output_shape(self):
        output = model(self.X, self.w_1, self.b_1, self.w_2, self.b_2)
        self.assertEqual(output.shape, (self.X.shape[0], 1))


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
            w.zeroGrad()
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
            batch_loss += loss.arr
            training_loss.append(batch_loss)

            w = w - self.lr * w.grad / self.m  # Correct weight update

        vectorized_time = time.time() - t
        print("Vectorized Training took (secs):", vectorized_time)

        # --- Convergence Checks ---
        print(f"Initial loss: {self.initial_loss}")
        print(f"Final loss: {batch_loss}")
        self.assertLess(batch_loss, self.initial_loss * 0.1)

        print(f"Estimated w: {w.arr.flatten()}, True w: {self.w_true}")
        self.assertTrue(np.allclose(w.arr.flatten(), self.w_true, atol=self.tolerance))
        self.assertLess(training_loss[-1], training_loss[0])
        self.assertLess(
            training_loss[int(len(training_loss) * 0.9)],
            training_loss[int(len(training_loss) * 0.1)],
        )

        return vectorized_time

    @pytest.mark.slow
    def test_convergence_non_vectorized(self):
        """Tests the non-vectorized implementation."""

        w0 = Tensor(0.0, requires_grad=True)
        w1 = Tensor(0.0, requires_grad=True)
        w2 = Tensor(0.0, requires_grad=True)

        training_loss = []
        t = time.time()

        for epoch in range(self.epochs):
            w0.zeroGrad()
            w1.zeroGrad()
            w2.zeroGrad()

            batch_loss = 0
            for x, y in zip(self.XX, self.YY):
                out = (
                    w0
                    + w1 * Tensor(x, requires_grad=False)
                    + w2 * Tensor(x, requires_grad=False) ** 2
                )
                loss = (out - Tensor(y, requires_grad=False)) ** 2
                loss.backprop()
                batch_loss += loss.arr

            training_loss.append(batch_loss)

            w0 = w0 - self.lr * w0.grad / self.m
            w1 = w1 - self.lr * w1.grad / self.m
            w2 = w2 - self.lr * w2.grad / self.m

        non_vectorized_time = time.time() - t
        print("Non-Vectorized Training took (secs):", non_vectorized_time)

        # --- Convergence Checks ---
        print(f"Initial loss: {self.initial_loss}")  # Use pre-computed initial loss
        print(f"Final loss: {batch_loss}")
        self.assertLess(batch_loss, self.initial_loss * 0.1)

        print(f"Estimated w0: {w0.arr}, True w0: {self.w_true[0]}")
        print(f"Estimated w1: {w1.arr}, True w1: {self.w_true[1]}")
        print(f"Estimated w2: {w2.arr}, True w2: {self.w_true[2]}")
        self.assertTrue(np.allclose(w0.arr, self.w_true[0], atol=self.tolerance))
        self.assertTrue(np.allclose(w1.arr, self.w_true[1], atol=self.tolerance))
        self.assertTrue(np.allclose(w2.arr, self.w_true[2], atol=self.tolerance))

        self.assertLess(training_loss[-1], training_loss[0])
        self.assertLess(
            training_loss[int(len(training_loss) * 0.9)],
            training_loss[int(len(training_loss) * 0.1)],
        )
        return non_vectorized_time

    @pytest.mark.slow
    def test_vectorized_vs_non_vectorized(self):
        vectorized_time = self.test_convergence_vectorized()
        non_vectorized_time = self.test_convergence_non_vectorized()

        print(f"Speedup: {non_vectorized_time / vectorized_time:.2f}x")
        self.assertGreater(
            non_vectorized_time,
            vectorized_time,
            "Vectorized implementation should be faster",
        )


if __name__ == "__main__":
    unittest.main()
