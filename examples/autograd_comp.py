import autograd.numpy as np
from autograd import grad

from tomi_grad import Var  # Import your Var class

def my_function(A, B):
    return 2 * (A + B).sum()


a = Var(2)
b = Var(3)

c = a * b
d = c + 1.0

e = d.sum()

e.backprop()

print(f'{a.grad = }')
print(f'{b.grad = }')
print(f'{c.grad = }')
print(f'{d.grad = }')
print(f'{e.grad = }')
