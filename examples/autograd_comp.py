from macrograd import Tensor
from macrograd.engine import topo_sort, Graph

def softmax(x: Tensor):
    e_x = x.exp()
    return e_x / (e_x.sum(axis=1, keepdims=True))


a = Tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
b = softmax(a)

b = b.reshape((-1, 1))
b = b.sum()

a.graph.visualize()

exec_list = topo_sort(a.graph.nodes)
print(exec_list)
