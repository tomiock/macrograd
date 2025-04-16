from macrograd import Tensor
from macrograd.engine import topo_sort

a = Tensor([1, 1, 1])
b = Tensor([2, 2, 2])
c = Tensor(1)

d = a * b
e = d * c

a.graph.visualize()
