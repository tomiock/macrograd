from numpy import allclose, where
import unittest

import autograd.numpy as np
from autograd import grad

from macrograd import Tensor, Graph


# Define the function using autograd's NumPy
def my_function(x, y):
    return (x @ y).sum()


class TestAutogradEquivalence(unittest.TestCase):
    def test_matmul_sum(self):
        # Create the input arrays (using autograd's NumPy)
        a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
        b = np.array([[7, 8], [9, 10], [11, 12]], dtype=np.float64)

        # Calculate the gradient of z with respect to x and y using autograd
        grad_z_x = grad(my_function, 0)
        grad_z_y = grad(my_function, 1)
        gradient_x_autograd = grad_z_x(a, b)
        gradient_y_autograd = grad_z_y(a, b)

        g = Graph()

        # Calculate using custom framework
        a_var = Tensor(a, requires_grad=True, graph=g)
        b_var = Tensor(b, requires_grad=True, graph=g)
        z_var = (a_var @ b_var).sum()

        # forward pass for the whole graph
        g.realize()
        # the backward need to be called on a single tensor
        z_var.backprop()

        # Assert that the gradients are close
        self.assertTrue(allclose(a_var.grad, gradient_x_autograd))
        self.assertTrue(allclose(b_var.grad, gradient_y_autograd))

    def test_matmul_sum_param(self):
        # Create the input arrays (using autograd's NumPy)
        a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
        b = np.array([[7, 8], [9, 10], [11, 12]], dtype=np.float64)

        # Calculate the gradient of z with respect to x and y using autograd
        grad_z_x = grad(my_function, 0)
        gradient_x_autograd = grad_z_x(a, b)

        g = Graph()

        # Calculate using custom framework
        a_var = Tensor(a, requires_grad=True, graph=g)
        b_var = Tensor(
            b, requires_grad=False, graph=g
        )  # this would be a parameters of the model
        z_var = (a_var @ b_var).sum()

        g.realize()
        z_var.backprop()

        # Assert that the gradients are close
        self.assertTrue(allclose(a_var.grad, gradient_x_autograd))

    def test_addition(self):
        a = np.array([[1.0, 2.0], [3.0, 4.0]])
        b = np.array([[5.0, 6.0], [7.0, 8.0]])

        grad_x_autograd = grad(lambda x, y: (x + y).sum(), 0)(a, b)
        grad_y_autograd = grad(lambda x, y: (x + y).sum(), 1)(a, b)

        g = Graph()

        a_var = Tensor(a.tolist(), requires_grad=True, graph=g)
        b_var = Tensor(b.tolist(), requires_grad=True, graph=g)
        z_var = (a_var + b_var).sum()

        g.realize()
        z_var.backprop()

        self.assertTrue(allclose(a_var.grad, grad_x_autograd))
        self.assertTrue(allclose(b_var.grad, grad_y_autograd))

    def test_subtraction(self):
        a = np.array([[1.0, 2.0], [3.0, 4.0]])
        b = np.array([[5.0, 6.0], [7.0, 8.0]])

        grad_x_autograd = grad(lambda x, y: (x - y).sum(), 0)(a, b)
        grad_y_autograd = grad(lambda x, y: (x - y).sum(), 1)(a, b)

        g = Graph()

        a_var = Tensor(a.tolist(), requires_grad=True, graph=g)
        b_var = Tensor(b.tolist(), requires_grad=True, graph=g)
        z_var = (a_var - b_var).sum()

        g.realize()
        z_var.backprop()

        self.assertTrue(allclose(a_var.grad, grad_x_autograd))
        self.assertTrue(allclose(b_var.grad, grad_y_autograd))

    def test_multiply(self):
        a = np.array([[1.0, 2.0], [3.0, 4.0]])
        b = np.array([[5.0, 6.0], [7.0, 8.0]])

        grad_x_autograd = grad(lambda x, y: (x * y).sum(), 0)(a, b)
        grad_y_autograd = grad(lambda x, y: (x * y).sum(), 1)(a, b)

        g = Graph()

        a_var = Tensor(a.tolist(), requires_grad=True, graph=g)
        b_var = Tensor(b.tolist(), requires_grad=True, graph=g)
        z_var = (a_var * b_var).sum()

        g.realize()
        z_var.backprop()

        self.assertTrue(np.allclose(a_var.grad, grad_x_autograd))
        self.assertTrue(np.allclose(b_var.grad, grad_y_autograd))

    def test_power_1(self):
        x_np = np.array([[1.0, 2.0], [3.0, 4.0]])
        y_np = np.array(2.0)

        g = Graph()

        x_var = Tensor(x_np.tolist(), requires_grad=True, graph=g)
        y_var = Tensor(y_np.tolist(), requires_grad=True, graph=g)
        z_var = (x_var**y_var).sum()

        g.realize()
        z_var.backprop()

        grad_x_autograd = grad(lambda x, y: (x**y).sum(), 0)(x_np, y_np)
        grad_y_autograd = grad(lambda x, y: (x**y).sum(), 1)(x_np, y_np)

        self.assertTrue(allclose(x_var.grad, grad_x_autograd))
        self.assertTrue(allclose(y_var.grad, grad_y_autograd))

    def test_transpose(self):
        x_np = np.array([[1.0, 2.0], [3.0, 4.0]])

        g = Graph()

        x_var = Tensor(x_np.tolist(), requires_grad=True, graph=g)
        z_var = x_var.T.sum()

        g.realize()
        z_var.backprop()

        grad_x_autograd = grad(lambda x: x.T.sum(), 0)(x_np)
        self.assertTrue(np.allclose(x_var.grad, grad_x_autograd))

    def test_sqrt(self):
        x_np = np.array([1.0, 4.0, 9.0, 16.0])

        g = Graph()

        x_var = Tensor(x_np.tolist(), requires_grad=True, graph=g)
        sqrt_var = x_var.sqrt()
        z_var = sqrt_var.sum()

        g.realize()
        z_var.backprop()

        grad_x_autograd = grad(lambda x: np.sqrt(x).sum())(x_np)
        self.assertTrue(np.allclose(x_var.grad, grad_x_autograd))

    def test_sum_axis(self):
        a_np = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)  # (2, 3)

        # Reference gradient calculation
        # Summing over axis 1 results in shape (2,) -> sum results -> scalar
        grad_a_autograd = grad(lambda x: x.sum(axis=1).sum(), 0)(a_np)

        # Your framework calculation
        g = Graph()
        a_var = Tensor(a_np.tolist(), requires_grad=True, graph=g)
        # Assuming Tensor.sum() maps to Ops.SUM and takes axis/keepdims
        sum_res = a_var.sum(axis=1, keepdims=False)  # Result shape (2,)
        z_var = sum_res.sum()  # Result shape ()

        g.realize()
        z_var.backprop()

        self.assertTrue(allclose(a_var.grad, grad_a_autograd))

    def test_sum_keepdims(self):
        a_np = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)  # (2, 3)

        # Reference gradient calculation
        # Summing over axis 1 with keepdims=True results in shape (2, 1) -> sum -> scalar
        grad_a_autograd = grad(lambda x: x.sum(axis=1, keepdims=True).sum(), 0)(a_np)

        # Your framework calculation
        g = Graph()
        a_var = Tensor(a_np.tolist(), requires_grad=True, graph=g)
        sum_res = a_var.sum(axis=1, keepdims=True)  # Result shape (2, 1)
        z_var = sum_res.sum()  # Result shape ()

        g.realize()
        z_var.backprop()

        self.assertTrue(allclose(a_var.grad, grad_a_autograd))

    def test_broadcast_add(self):
        a_np = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)  # (2, 3)
        b_np = np.array([10.0, 20.0, 30.0], dtype=np.float64)  # (3,) - broadcastable

        # Reference gradient calculation
        grad_a_autograd = grad(lambda x, y: (x + y).sum(), 0)(a_np, b_np)
        grad_b_autograd = grad(lambda x, y: (x + y).sum(), 1)(a_np, b_np)

        # Your framework calculation
        g = Graph()
        a_var = Tensor(a_np.tolist(), requires_grad=True, graph=g)
        b_var = Tensor(b_np.tolist(), requires_grad=True, graph=g)
        z_var = (a_var + b_var).sum()  # Broadcasting happens in '+'

        g.realize()
        z_var.backprop()

        print(z_var.data)

        print(f"{b_var.grad = }")
        print(f"{grad_b_autograd = }")
        print()

        print(f"{a_var.grad = }")
        print(f"{grad_a_autograd = }")

        self.assertTrue(allclose(a_var.grad, grad_a_autograd))
        self.assertTrue(allclose(b_var.grad, grad_b_autograd))

    def test_broadcast_mul(self):
        a_np = np.array(
            [[1.0], [2.0], [3.0]], dtype=np.float64
        )  # (3, 1) - broadcastable
        b_np = np.array([10.0, 20.0, 30.0], dtype=np.float64)  # (3,)

        # Reference gradient calculation
        grad_a_autograd = grad(lambda x, y: (x * y).sum(), 0)(a_np, b_np)
        grad_b_autograd = grad(lambda x, y: (x * y).sum(), 1)(a_np, b_np)

        # Your framework calculation
        g = Graph()
        a_var = Tensor(a_np.tolist(), requires_grad=True, graph=g)
        b_var = Tensor(b_np.tolist(), requires_grad=True, graph=g)
        z_var = (a_var * b_var).sum()  # Broadcasting happens in '*'

        g.realize()
        z_var.backprop()

        self.assertTrue(allclose(a_var.grad, grad_a_autograd))
        self.assertTrue(allclose(b_var.grad, grad_b_autograd))

    def test_batched_matmul(self):
        a_np = np.arange(2, dtype=np.float64).reshape(2, 1)
        b_np = np.arange(16, dtype=np.float64).reshape(8, 2, 1)

        # Your framework calculation
        g = Graph()
        a_var = Tensor(a_np.tolist(), requires_grad=True, graph=g)
        b_var = Tensor(b_np.tolist(), requires_grad=True, graph=g)
        z_var = (a_var.T @ b_var).sum()  # Uses batched matmul

        g.realize()

        z_var.backprop()

        # Reference gradient calculation
        grad_a_autograd = grad(lambda x, y: (x.T @ y).sum(), 0)(a_np, b_np)
        grad_b_autograd = grad(lambda x, y: (x.T @ y).sum(), 1)(a_np, b_np)

        self.assertTrue(allclose(a_var.grad, grad_a_autograd))
        self.assertTrue(allclose(b_var.grad, grad_b_autograd))

    def test_exp(self):
        # Requires Ops.EXP and its gradient (VJP: dL/dx = dL/dy * exp(x))
        a_np = np.array([0.0, 1.0, -2.0, 3.0], dtype=np.float64)

        grad_a_autograd = grad(lambda x: np.exp(x).sum(), 0)(a_np)

        g = Graph()
        a_var = Tensor(a_np.tolist(), requires_grad=True, graph=g)
        z_var = a_var.exp().sum()  # Assumes .exp() method exists and maps to Ops.EXP

        g.realize()
        z_var.backprop()

        self.assertTrue(allclose(a_var.grad, grad_a_autograd))

    def test_log(self):
        # Requires Ops.LOG and its gradient (VJP: dL/dx = dL/dy * (1/x))
        # Use positive values for log
        a_np = np.array([1.0, 2.0, 0.5, 10.0], dtype=np.float64)

        grad_a_autograd = grad(lambda x: np.log(x).sum(), 0)(a_np)

        g = Graph()
        a_var = Tensor(a_np.tolist(), requires_grad=True, graph=g)
        z_var = a_var.log().sum()  # Assumes .log() method exists (natural log)

        g.realize()
        z_var.backprop()

        self.assertTrue(allclose(a_var.grad, grad_a_autograd))

    def test_log_base_2(self):
        a_np = np.array([1.0, 2.0, 0.5, 10.0], dtype=np.float64)

        base = 2
        grad_a_autograd = grad(lambda x: np.log2(x).sum(), 0)(a_np)

        g = Graph()
        a_var = Tensor(a_np.tolist(), requires_grad=True, graph=g)
        z_var = a_var.log(base=base).sum()  # Assumes .log() method exists (natural log)

        g.realize()
        z_var.backprop()

        self.assertTrue(allclose(a_var.grad, grad_a_autograd))

    def test_reshape(self):
        # Requires Ops.RESHAPE and its gradient (VJP: dL/dx = reshape(dL/dy, original_shape))
        a_np = np.arange(12, dtype=np.float64).reshape(3, 4)
        target_shape = (2, 6)

        grad_a_autograd = grad(lambda x: x.reshape(target_shape).sum(), 0)(a_np)

        g = Graph()
        a_var = Tensor(a_np.tolist(), requires_grad=True, graph=g)
        # Assumes .reshape() method exists and maps to Ops.RESHAPE
        z_var = a_var.reshape(target_shape).sum()

        g.realize()
        z_var.backprop()

        self.assertTrue(allclose(a_var.grad, grad_a_autograd))
        self.assertEqual(a_var.grad.shape, a_np.shape)  # Check grad shape

    def test_max_axis(self):
        # Requires Ops.MAX and its gradient (grad_max function)
        a_np = np.array([[1.0, 5.0, 2.0], [4.0, 3.0, 6.0]], dtype=np.float64)
        axis_to_reduce = 1

        grad_a_autograd = grad(lambda x: x.max(axis=axis_to_reduce).sum(), 0)(a_np)
        print(grad_a_autograd)

        g = Graph()
        a_var = Tensor(a_np.tolist(), requires_grad=True, graph=g)
        # Assumes .max() method exists
        z_var = a_var.max(axis=axis_to_reduce, keepdims=False).sum()

        g.realize()
        z_var.backprop()

        print(a_var.grad)

        self.assertTrue(
            allclose(a_var.grad, grad_a_autograd),
            f"Framework:{a_var.grad}\n  Autograd: {grad_a_autograd}\n")

    def test_gradient_accumulation(self):
        a_np = np.array(2.0, dtype=np.float64)
        b_np = np.array(3.0, dtype=np.float64)
        c_np = np.array(4.0, dtype=np.float64)

        # Function: loss = (a*b) + (a*c). dL/da = b + c
        grad_a_autograd = grad(lambda a, b, c: (a * b) + (a * c), 0)(a_np, b_np, c_np)

        g = Graph()
        a_var = Tensor(a_np.tolist(), requires_grad=True, graph=g)
        b_var = Tensor(
            b_np.tolist(), requires_grad=False, graph=g
        )  # Treat b, c as constants
        c_var = Tensor(c_np.tolist(), requires_grad=False, graph=g)

        y = a_var * b_var  # First use of a_var
        z = a_var * c_var  # Second use of a_var
        loss = y + z  # Add results. Loss depends on 'a' through two paths.

        g.realize()
        loss.backprop()

        # Check if the gradient for 'a' was correctly accumulated
        self.assertTrue(allclose(a_var.grad, grad_a_autograd))
        self.assertEqual(a_var.grad, b_np + c_np)  # Check exact value

    def test_power_scalar_exponent(self):
        # Test specifically using a scalar exponent in the call
        x_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        exponent = 3.0  # Python scalar float

        grad_x_autograd = grad(lambda x: (x**exponent).sum(), 0)(x_np)
        # Autograd doesn't compute grad w.r.t Python scalars

        g = Graph()
        x_var = Tensor(x_np.tolist(), requires_grad=True, graph=g)
        # Use the overloaded operator with a Python scalar
        z_var = (x_var**exponent).sum()

        g.realize()
        z_var.backprop()

        self.assertTrue(allclose(x_var.grad, grad_x_autograd))

    def test_relu(self):
        """Tests the gradient calculation for the ReLU operation."""
        # Input data including positive, negative, and zero values
        a_np = np.array(
            [[-2.0, -0.1, 0.0],
             [1.0, 2.5, -0.0],
             [3.0, -3.0, 4.0]], dtype=np.float64
        )

        # Define the function using autograd.numpy.maximum for ReLU
        # We sum the result to get a scalar output for autograd.grad
        relu_sum_func = lambda x: np.maximum(0, x).sum()

        # Calculate the reference gradient using autograd
        grad_a_autograd = grad(relu_sum_func, 0)(a_np)
        # Expected gradient based on dy/dx = (x > 0):
        # [[0., 0., .5],
        #  [1., 1., .5],
        #  [1., 0., 1.]]
        grad_autograd_modified = grad_a_autograd.copy()
        point_5_mask = (grad_a_autograd == .5)
        grad_autograd_modified[point_5_mask] = 0.

        g = Graph()  # New graph for the test
        a_var = Tensor(a_np.tolist(), requires_grad=True, graph=g)

        z_var = a_var.relu().sum()

        g.realize()
        z_var.backprop()  # Compute gradients

        # --- Assertions ---
        self.assertIsNotNone(a_var.grad, "Gradient for input 'a' should not be None")
        self.assertEqual(a_var.grad.shape, grad_autograd_modified.shape, "Gradient shape mismatch")
        self.assertTrue(
            allclose(a_var.grad, grad_autograd_modified),
            f"ReLU Gradient mismatch:\nFramework:\n{a_var.grad}\nAutograd:\n{grad_a_autograd}",
        )


if __name__ == "__main__":
    unittest.main()
