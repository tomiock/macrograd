from macrograd import Tensor
from macrograd.engine import topo_sort

a = Tensor([[1, 1, 1, 1]])

c = a.sqrt()

a.graph.visualize()
