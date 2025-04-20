from macrograd import Tensor, Graph

from numpy.random import randn

from macrograd.engine import topo_sort

g = Graph()

data = Tensor(randn(2,).tolist(), graph=g)
weight = Tensor(randn(2, 2).tolist(), graph=g, requires_grad=True)
bias = Tensor(randn(2,).tolist(), graph=g, requires_grad=True)

x = (data @ weight) + bias
g.realize()

x.backprop()
