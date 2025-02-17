import numpy as np
import time
import unittest
from tomi_grad.tensor import Var


class TestPolynomialRegression(unittest.TestCase):
    def test_convergence(self):
        m = 100  # number of training points
        w_true = [-1.0, 2.0, 1.0]  # True weights

        XX = 4 * np.random.rand(m) - 2
        YY = w_true[0] + w_true[1] * XX + w_true[2] * (XX**2)
        YY += np.random.normal(scale=0.3, size=YY.size)

        # Initialize weights
        w0 = Var(0.0, requires_grad=True)
        w1 = Var(0.0, requires_grad=True)
        w2 = Var(0.0, requires_grad=True)

        lr = 0.01
        epochs = 200
        training_loss = []

        t = time.time()

        for _ in range(epochs):
            w0.zeroGrad()
            w1.zeroGrad()
            w2.zeroGrad()

            batch_loss = 0
            for x, y in zip(XX, YY):
                out = w0 + w1 * x + w2 * (x**2)
                loss = (out - y) ** 2
                loss.backprop()  # Use backprop
                batch_loss += loss.arr

            training_loss.append(batch_loss)

            w0.arr = w0.arr - lr * w0.grad / m
            w1.arr = w1.arr - lr * w1.grad / m
            w2.arr = w2.arr - lr * w2.grad / m

        print("Training took (secs):", time.time() - t)

        # --- Convergence Checks ---

        # 1. Check if the final loss is reasonably low.  The exact value will
        #    depend on the noise level, but it should be significantly smaller
        #    than the initial loss.  We'll check against an *initial* loss.
        initial_loss = 0
        for x, y in zip(XX, YY):
            out = Var(0.0) + Var(0.0) * x + Var(0.0) * (x**2)  # Initial w0, w1, w2
            loss = (out - y) ** 2
            initial_loss += loss.arr

        print(f"Initial loss: {initial_loss}")
        print(f"Final loss: {batch_loss}")
        self.assertLess(batch_loss, initial_loss * 0.1)  # Check for a good decrease

        # 2. Check if the estimated weights are close to the true weights.
        tolerance = 0.5  # Adjust as needed
        print(f"Estimated w0: {w0.arr}, True w0: {w_true[0]}")
        print(f"Estimated w1: {w1.arr}, True w1: {w_true[1]}")
        print(f"Estimated w2: {w2.arr}, True w2: {w_true[2]}")
        self.assertTrue(np.allclose(w0.arr, w_true[0], atol=tolerance))
        self.assertTrue(np.allclose(w1.arr, w_true[1], atol=tolerance))
        self.assertTrue(np.allclose(w2.arr, w_true[2], atol=tolerance))

        # 3. Check that the training loss decreased (monotonically decreasing is ideal,
        # but some fluctuation is okay due to the stochastic nature)
        # We check that the final loss is less than the initial AND
        # the loss at 90% is lower than at 10%
        self.assertLess(training_loss[-1], training_loss[0])  # Final lower than initial
        self.assertLess(
            training_loss[int(len(training_loss) * 0.9)],
            training_loss[int(len(training_loss) * 0.1)],
        )  # Loss at 90% should be lower than at 10%.


if __name__ == "__main__":
    unittest.main()
