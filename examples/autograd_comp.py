from numpy import require
from macrograd import Tensor
from macrograd.tensor import get_formula, trace_forward, get_graph

def my_function_bias(x, w, b):
    return (x @ w) + b
 
def my_function(x, w):
    return (x @ w)

x = Tensor([1, 2], requires_grad=False)
w = Tensor([2, 1], requires_grad=True)
b = Tensor([3], requires_grad=True)

z = my_function_bias(x, w, b)

get_formula(z)
