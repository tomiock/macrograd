from macrograd import Tensor
from macrograd.engine import topo_sort

import numpy as np

e = Tensor(np.e)

def softmax(x: Tensor):
    e_x = e**x
    return e_x / (e_x.sum(axis=1, keepdims=True))


a = Tensor([1, 2, 3, 4])
b = softmax(a)

a.graph.visualize()
