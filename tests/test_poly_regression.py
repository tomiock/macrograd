import pytest
import numpy as np
import time
import unittest
from tomi_grad.tensor import Tensor

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
        self.YY = self.w_true[0] + self.w_true[1] * self.XX + self.w_true[2] * (self.XX ** 2)
        self.YY += np.random.normal(scale=0.3, size=self.YY.size)
        self.lr = 0.01
        self.epochs = 200
        self.tolerance = 0.5

        # Calculate initial loss (vectorized)
        X = np.stack([np.ones(self.m), self.XX, self.XX**2], axis=1)
        self.initial_loss = np.sum((X @ np.array([0,0,0]).reshape(-1,1) - self.YY.reshape(-1,1))**2)


    def test_convergence_vectorized(self):

        w = Tensor(np.random.normal(size=(3, 1)), requires_grad=True)  # 3x1 vector
        training_loss = []

        t = time.time()

        for _ in range(self.epochs):
            w.zeroGrad()
            batch_loss = 0

            # Vectorized loss calculation
            X = Tensor(np.stack([np.ones(self.m), self.XX, self.XX**2], axis=1), requires_grad=False) # m x 3
            Y = Tensor(self.YY.reshape(-1,1), requires_grad=False)  # Convert Y to a column vector (m x 1)

            out = X @ w  # (m x 3) @ (3 x 1) -> (m x 1)
            loss = ((out - Y) ** 2).sum()

            loss.backprop()
            batch_loss += loss.arr
            training_loss.append(batch_loss)

            w = w - self.lr * w.grad / self.m # Correct weight update

        vectorized_time = time.time() - t
        print("Vectorized Training took (secs):", vectorized_time)

        # --- Convergence Checks ---
        print(f"Initial loss: {self.initial_loss}")
        print(f"Final loss: {batch_loss}")
        self.assertLess(batch_loss, self.initial_loss * 0.1)

        print(f"Estimated w: {w.arr.flatten()}, True w: {self.w_true}")
        self.assertTrue(np.allclose(w.arr.flatten(), self.w_true, atol=self.tolerance))
        self.assertLess(training_loss[-1] , training_loss[0])
        self.assertLess(training_loss[int(len(training_loss)*0.9)], training_loss[int(len(training_loss)*0.1)])

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
                out = w0 + w1 * Tensor(x, requires_grad=False) + w2 * Tensor(x, requires_grad=False)**2
                loss = (out - Tensor(y, requires_grad=False))**2
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

        self.assertLess(training_loss[-1] , training_loss[0])
        self.assertLess(training_loss[int(len(training_loss)*0.9)], training_loss[int(len(training_loss)*0.1)])
        return non_vectorized_time

    @pytest.mark.slow
    def test_vectorized_vs_non_vectorized(self):

        vectorized_time = self.test_convergence_vectorized()
        non_vectorized_time = self.test_convergence_non_vectorized()

        print(f"Speedup: {non_vectorized_time / vectorized_time:.2f}x")
        self.assertGreater(non_vectorized_time, vectorized_time, "Vectorized implementation should be faster")

if __name__ == '__main__':
    unittest.main()
