from macrograd import Tensor
from macrograd.tensor import get_axes_broadcasting

def my_function(A, B):
    return 2 * (A + B).sum()


a = Tensor([2, 3], requires_grad=1)
b = Tensor([4, 5], requires_grad=1)

c = a * b
c.backprop()

print(f'{a.grad = }')

sum_axes = get_axes_broadcasting(a.grad, a.shape)
print(sum_axes)
