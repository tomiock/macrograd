import autograd.numpy as np
from autograd import grad
import unittest
from tomi_grad import Var


# Define the function using autograd's NumPy
def my_function(x, y):
    return (x @ y).sum()


class TestAutogradEquivalence(unittest.TestCase):
    def test_matmul_sum(self):
        # Create the input arrays (using autograd's NumPy)
        a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
        b = np.array([[7, 8], [9, 10], [11, 12]], dtype=np.float64)

        # Calculate the gradient of z with respect to x and y using autograd
        grad_z_x = grad(my_function, argnum=0)
        grad_z_y = grad(my_function, argnum=1)
        gradient_x_autograd = grad_z_x(a, b)
        gradient_y_autograd = grad_z_y(a, b)

        # Calculate using custom framework
        a_var = Var(a)
        b_var = Var(b)
        z_var = (a_var @ b_var).sum()
        z_var._backward()  # Use _backward for custom framework

        # Assert that the gradients are close
        self.assertTrue(np.allclose(a_var.grad, gradient_x_autograd))
        self.assertTrue(np.allclose(b_var.grad, gradient_y_autograd))
        print("Test Matmul + Sum Passed")

    def test_addition(self):
        a = np.array([[1.0, 2.0], [3.0, 4.0]])
        b = np.array([[5.0, 6.0], [7.0, 8.0]])

        grad_x_autograd = grad(lambda x, y: (x + y).sum(), argnum=0)(a, b)
        grad_y_autograd = grad(lambda x, y: (x + y).sum(), argnum=1)(a, b)

        a_var = Var(a.tolist())
        b_var = Var(b.tolist())
        z_var = (a_var + b_var).sum()
        z_var._backward()
        self.assertTrue(np.allclose(a_var.grad, grad_x_autograd))
        self.assertTrue(np.allclose(b_var.grad, grad_y_autograd))
        print("Test Addition Passed")

    def test_subtraction(self):
        a = np.array([[1.0, 2.0], [3.0, 4.0]])
        b = np.array([[5.0, 6.0], [7.0, 8.0]])

        grad_x_autograd = grad(lambda x, y: (x - y).sum(), argnum=0)(a, b)
        grad_y_autograd = grad(lambda x, y: (x - y).sum(), argnum=1)(a, b)

        a_var = Var(a.tolist())
        b_var = Var(b.tolist())
        z_var = (a_var - b_var).sum()
        z_var._backward()
        self.assertTrue(np.allclose(a_var.grad, grad_x_autograd))
        self.assertTrue(np.allclose(b_var.grad, grad_y_autograd))
        print("Test Substraction Passed")
    
    def test_multiplication(self):
        a = np.array([[1.0, 2.0], [3.0, 4.0]])
        b = np.array([[5.0, 6.0], [7.0, 8.0]])

        grad_x_autograd = grad(lambda x, y: (x * y).sum(), argnum=0)(a, b)
        grad_y_autograd = grad(lambda x, y: (x * y).sum(), argnum=1)(a, b)

        a_var = Var(a.tolist())
        b_var = Var(b.tolist())
        z_var = (a_var * b_var).sum()
        z_var._backward()
        self.assertTrue(np.allclose(a_var.grad, grad_x_autograd))
        self.assertTrue(np.allclose(b_var.grad, grad_y_autograd))
        print("Test Multiplication Passed")
    def test_power(self):        
        x_np = np.array([[1.0, 2.0], [3.0, 4.0]])
        y_np = np.array([2.0, 3.0])
        x_var = Var(x_np.tolist(), requires_grad=True)
        y_var = Var(y_np.tolist(), requires_grad=True)
        z_var = (x_var ** y_var).sum()
        z_var._backward()

        grad_x_autograd = grad(lambda x, y: (x ** y).sum(), argnum=0)(x_np, y_np)
        grad_y_autograd = grad(lambda x, y: (x ** y).sum(), argnum=1)(x_np, y_np)
        self.assertTrue(np.allclose(x_var.grad, grad_x_autograd))
        self.assertTrue(np.allclose(y_var.grad, grad_y_autograd))
        print("Test Power Passed")

    def test_transpose(self):
        x_np = np.array([[1.0, 2.0], [3.0, 4.0]])
        x_var = Var(x_np.tolist(), requires_grad = True)

        z_var = x_var.T.sum()
        z_var._backward()

        grad_x_autograd = grad(lambda x: x.T.sum(), argnum=0)(x_np)
        self.assertTrue(np.allclose(x_var.grad, grad_x_autograd))
        print("Test Transpose Passed")


if __name__ == '__main__':
    unittest.main()
