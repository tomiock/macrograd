from macrograd import Tensor
from macrograd.engine import topo_sort

a = Tensor([1, 1, 1])
b = Tensor([2, 2, 2])

c = a + b
d = c / 10
e = d @ d
f = e.T
h = f.sqrt()
i = h.sum()
j = i.reshape(1, 1)
k = j + a


exec_list = topo_sort(j.graph.nodes)
print(exec_list)

k.graph.visualize()
