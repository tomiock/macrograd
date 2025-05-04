from macrograd import Tensor, Graph

from numpy.random import randn
import numpy as np

from macrograd.engine import topo_sort


data_array = randn(2, 10)
weight_array = randn(2, 2)
bias_array = randn(2,1)

for i in range(10):
    g = Graph()
    data = Tensor(data_array.tolist(), graph=g)
    weight = Tensor(weight_array.tolist(), graph=g, requires_grad=True)
    bias = Tensor(bias_array.tolist(), graph=g, requires_grad=True)

    x = bias + (weight @ data)
    g.realize()

    x.backprop()
    print(g)
