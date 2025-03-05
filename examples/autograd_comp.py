from macrograd import Tensor
from macrograd.tensor import get_axes_broadcasting

def my_function_bias(x, w, b):
    return (x @ w) + b
 
def my_function(x, w):
    return (x @ w)

x = Tensor([1, 2])
w = Tensor([2, 1], requires_grad=True)
b = Tensor([3], requires_grad=True)

z = my_function_bias(x, w, b)

n, e = z._trace()
print(n)
print(e)
