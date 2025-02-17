from tomi_grad.tensor import Tensor
from tomi_grad.graph import visualize_graph

# Create a simple graph
a = Tensor([[1, 2], [3, 4]], requires_grad=True)
b = Tensor([[5, 6], [7, 8]], requires_grad=True)
c = a * b
d = c.sum()
visualize_graph(d)
